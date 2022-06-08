//===- LinalgMapToTpp.cpp ----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/MathxOps.h"
#include "Standalone/TppOps.h"
#include "Standalone/TppPasses.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "Standalone/TppPasses.h.inc"

#define DEBUG_TYPE "linalg-map-to-tpp"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

namespace {

struct MapGenericOpToTpp : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  bool hasOnlyProjectedPermutations(linalg::GenericOp linalgOp) const {
    return llvm::all_of(linalgOp.getIndexingMaps(), [](AffineMap m) {
      return m.isProjectedPermutation(/*allowZeroInResults=*/true);
    });
  }

  // Return true if the operation is an element-wise linalg op.
  bool isElementWise(linalg::GenericOp linalgOp) const {
    if (linalgOp.getNumLoops() != linalgOp.getNumParallelLoops())
      return false;
    if (!hasOnlyProjectedPermutations(linalgOp))
      return false;
    for (OpOperand *operand : linalgOp.getOutputOperands())
      if (!linalgOp.getTiedIndexingMap(operand).isPermutation())
        return false;
    return true;
  }

  // Return true if: 1) the region has a single block. 2) The block has two
  // operations only (linalg.YieldOp and OP). 3) The operation result types are
  // int or float.
  // TODO: For now we assume the region to have only two operations: The YieldOp
  // and the 'OP', meaning that the entire linalg.generic will map to a single
  // tpp operation. If we do element-wise fusion at the linalg level this
  // assumption does not hold anymore as now a linalg.generic can map to n tpp
  // operations. If we support 1:n matching what should we do if the entire
  // linalg.op cannot be replace by tpp operations?
  template <typename OP>
  bool hasOnlyScalarElementwiseOp(Region &region) const {
    if (!region.hasOneBlock())
      return false;
    if (std::distance(region.front().begin(), region.front().end()) != 2)
      return false;
    for (Operation &op : region.front()) {
      if (!isa<OP, linalg::YieldOp>(op) ||
          llvm::any_of(op.getResultTypes(),
                       [](Type type) { return !type.isIntOrFloat(); }))
        return false;
    }
    return true;
  }

  // Return true if the linalgOp contains only the yieldOp.
  bool hasOnlyYieldOp(Region &region) const {
    if (!region.hasOneBlock())
      return false;
    return std::distance(region.front().begin(), region.front().end()) == 1;
  }

  // Return true if the operation can maps to a tpp operation. The generic
  // must have 'numOperands' operands, 'numResults' result at buffer level
  // while 'numOperands' operands, 'numResults' + 1 result at tensor level.
  // In both cases, the operands type are expected to be 2-dimensional shaped
  // type. Why this?
  // - The mapping to tpp should work both at the tensor and buffer level.
  // If you consider an element-wise operation, it has the following properties:
  // a) At buffer level it has 0 results and 2 operands.
  // b) At tensor level it produces 1 result and takes 2 operands.
  bool canMapToTppImpl(linalg::GenericOp linalgOp, unsigned numResults,
                       unsigned numOperands) const {
    if (linalgOp.hasBufferSemantics())
      if ((linalgOp->getNumResults() != numResults) ||
          (linalgOp->getNumOperands() != numOperands))
        return false;
    if (linalgOp.hasTensorSemantics())
      if ((linalgOp->getNumResults() != numResults + 1) ||
          (linalgOp->getNumOperands() != numOperands))
        return false;

    for (unsigned idx = 0; idx < numOperands; idx++) {
      if (ShapedType operandType = linalgOp->getOperand(idx)
                                       .getType()
                                       .dyn_cast_or_null<ShapedType>())
        if (!operandType.hasStaticShape())
          return false;
    }
    return true;
  }

  bool canMapToTppUnary(linalg::GenericOp linalgOp) const {
    return canMapToTppImpl(linalgOp, /*numResults*/ 0, /*numOperands*/ 2);
  }

  bool canMapToTppBinary(linalg::GenericOp linalgOp) const {
    return canMapToTppImpl(linalgOp, /*numResults*/ 0, /*numOperands*/ 3);
  }

  // Return true if the linalg.generic maps to a tpp.gemm.
  bool isTPPGemm(linalg::GenericOp linalgOp) const {
    // structural and access pattern.
    ArrayAttr iteratorTypes = linalgOp.iterator_types();
    if (iteratorTypes.size() != 3)
      return false;
    if (!(isParallelIterator(iteratorTypes[0]) &&
          isParallelIterator(iteratorTypes[1]) &&
          isReductionIterator(iteratorTypes[2])))
      return false;
    using MapList = ArrayRef<ArrayRef<AffineExpr>>;
    auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
    AffineExpr i, j, k;
    bindDims(linalgOp.getContext(), i, j, k);
    if (linalgOp.getIndexingMaps() != infer({{i, k}, {k, j}, {i, j}}))
      return false;
    // operations and operands.
    Region &region = linalgOp.getRegion();
    if (!region.hasOneBlock() ||
        !canMapToTppImpl(linalgOp, /*numResults*/ 0, /*numOperands*/ 3))
      return false;
    using mlir::matchers::m_Any;
    using mlir::matchers::m_Val;
    Block &block = region.front();
    linalg::YieldOp yield = cast<linalg::YieldOp>(block.getTerminator());
    if (yield.getNumOperands() != 1)
      return false;
    if (block.getNumArguments() != 3)
      return false;
    // TODO: this low-tech stuff is too manual (see:
    // https://discourse.llvm.org/t/linalg-to-llvm-lowering/4867/7)
    Operation *maybeAdd = yield.getOperands()[0].getDefiningOp();
    auto mFloat =
        m_Op<arith::AddFOp>(m_Val(block.getArgument(2)),
                            m_Op<arith::MulFOp>(m_Val(block.getArgument(0)),
                                                m_Val(block.getArgument(1))));
    auto mInteger =
        m_Op<arith::AddIOp>(m_Val(block.getArgument(2)),
                            m_Op<arith::MulIOp>(m_Val(block.getArgument(0)),
                                                m_Val(block.getArgument(1))));
    return (mFloat.match(maybeAdd) || mInteger.match(maybeAdd));
  }

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (isElementWise(linalgOp)) {
      if (hasOnlyYieldOp(linalgOp.getRegion()) && canMapToTppUnary(linalgOp)) {
        StringAttr tppMicroKernelName = rewriter.getStringAttr("tpp.identity");
        rewriter.updateRootInPlace(
            linalgOp, [&]() { linalgOp.library_callAttr(tppMicroKernelName); });
        return success();
      }
      if (hasOnlyScalarElementwiseOp<mathx::ReluOp>(linalgOp.getRegion()) &&
          canMapToTppUnary(linalgOp)) {
        StringAttr tppMicroKernelName = rewriter.getStringAttr("tpp.relu");
        rewriter.updateRootInPlace(
            linalgOp, [&]() { linalgOp.library_callAttr(tppMicroKernelName); });
        return success();
      }
      if (hasOnlyScalarElementwiseOp<arith::AddFOp>(linalgOp.getRegion()) &&
          canMapToTppBinary(linalgOp)) {
        StringAttr tppMicroKernelName = rewriter.getStringAttr("tpp.add");
        rewriter.updateRootInPlace(
            linalgOp, [&]() { linalgOp.library_callAttr(tppMicroKernelName); });
        return success();
      }
    }
    if (isTPPGemm(linalgOp)) {
      StringAttr tppMicroKernelName = rewriter.getStringAttr("tpp.matmul");
      rewriter.updateRootInPlace(
          linalgOp, [&]() { linalgOp.library_callAttr(tppMicroKernelName); });
      return success();
    }
    return failure();
  }
};

void populateLinalgToTppPatterns(RewritePatternSet &patterns) {
  patterns.add<MapGenericOpToTpp>(patterns.getContext());
}

struct MapToTpp : public LinalgMapToTppBase<MapToTpp> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateLinalgToTppPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // end namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createMapLinalgToTppPass() {
  return std::make_unique<MapToTpp>();
}
