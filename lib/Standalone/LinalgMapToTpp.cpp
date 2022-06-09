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
#include "mlir/Dialect/Tensor/Utils/Utils.h"
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
    if (!region.hasOneBlock() || !hasStaticShape(linalgOp))
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

  // Return true if the operands of the linalg.generic have
  // static shapes.
  bool hasStaticShape(linalg::GenericOp linalgOp) const {
    return linalgOp.hasDynamicShape() != true;
  }

  // Ensure the SIMD dimension to be multiple of 16.
  // TODO: Should happen here? and a bit too specific..
  //
  // %0 = tensor.pad (%C) : tensor<3x3xf32> to tensor<3xSIMDxf32>
  // %1 = tensor.pad (%B) : tensor<3x3xf32> to tensor<3xSIMDxf32>
  // %2 = linalg.matmul(%C, %A, %B)
  // %3 = tensor.extract tensor<3xSIMDxf32> to tensor<3x3xf32>
  LogicalResult makeSIMDFriendly(linalg::GenericOp linalgOp,
                                 PatternRewriter &rewriter) const {
    if (!linalgOp.hasTensorSemantics() || !hasStaticShape(linalgOp))
      return failure();
    Location loc = linalgOp.getLoc();
    Value C = linalgOp->getOperand(2);
    Value B = linalgOp->getOperand(1);
    Value A = linalgOp->getOperand(0);

    ArrayRef<int64_t> shapeC = C.getType().cast<ShapedType>().getShape();
    ArrayRef<int64_t> shapeB = B.getType().cast<ShapedType>().getShape();
    assert(shapeC[1] == shapeB[1]);
    int64_t simdDim = shapeC[1];
    if (simdDim % 16 == 0)
      return success();
    int64_t paddedSimd = 16 * std::ceil((float)simdDim / 16.0);
    SmallVector<int64_t> newShapeC = {shapeC[0], paddedSimd};
    SmallVector<int64_t> newShapeB = {shapeB[0], paddedSimd};
    RankedTensorType newRankedC = RankedTensorType::get(
        newShapeC, C.getType().cast<ShapedType>().getElementType());
    RankedTensorType newRankedB = RankedTensorType::get(
        newShapeB, B.getType().cast<ShapedType>().getElementType());
    Value padZero = rewriter.create<arith::ConstantOp>(
        loc, C.getType().cast<ShapedType>().getElementType(),
        rewriter.getZeroAttr(C.getType().cast<ShapedType>().getElementType()));
    // Value padOne = rewriter.create<arith::ConstantOp>(
    //     linalgOp.getLoc(), C.getType().cast<ShapedType>().getElementType(),
    //     rewriter.getOneAttr(C.getType().cast<ShapedType>().getElementType()));
    Value paddedC = tensor::createPadHighOp(newRankedC, C, padZero,
                                            /*nofold*/ false, loc, rewriter);
    Value paddedB = tensor::createPadHighOp(newRankedB, B, padZero,
                                            /*nofold*/ false, loc, rewriter);

    linalg::GenericOp replacementOp = rewriter.create<linalg::GenericOp>(
        loc, paddedC.getType(), ValueRange{A, paddedB}, ValueRange{paddedC},
        linalgOp.getIndexingMaps(),
        llvm::to_vector<4>(
            linalgOp.iterator_types().template getAsValueRange<StringAttr>()));
    rewriter.inlineRegionBefore(linalgOp.region(), replacementOp.region(),
                                replacementOp.region().begin());

    unsigned rank = 2;
    SmallVector<OpFoldResult, 4> offsets, sizes, strides;
    offsets.reserve(rank);
    sizes.reserve(rank);
    strides.reserve(rank);
    for (unsigned r = 0; r < rank; r++) {
      offsets.push_back(rewriter.getIndexAttr(0));
      strides.push_back(rewriter.getIndexAttr(1));
      sizes.push_back(rewriter.getIndexAttr(shapeC[r]));
    }

    Value extract = rewriter.create<tensor::ExtractSliceOp>(
        loc, replacementOp->getResult(0), offsets, sizes, strides);

    // auto m = linalgOp->getParentOfType<ModuleOp>();
    // m.dump();
    // assert(0);
    rewriter.replaceOp(linalgOp, extract);
    return success();
  }

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (isElementWise(linalgOp)) {
      if (hasOnlyYieldOp(linalgOp.getRegion()) && hasStaticShape(linalgOp)) {
        StringAttr tppMicroKernelName = rewriter.getStringAttr("tpp.identity");
        rewriter.updateRootInPlace(
            linalgOp, [&]() { linalgOp.library_callAttr(tppMicroKernelName); });
        return success();
      }
      if (hasOnlyScalarElementwiseOp<mathx::ReluOp>(linalgOp.getRegion()) &&
          hasStaticShape(linalgOp)) {
        StringAttr tppMicroKernelName = rewriter.getStringAttr("tpp.relu");
        rewriter.updateRootInPlace(
            linalgOp, [&]() { linalgOp.library_callAttr(tppMicroKernelName); });
        return success();
      }
      if (hasOnlyScalarElementwiseOp<arith::AddFOp>(linalgOp.getRegion()) &&
          hasStaticShape(linalgOp)) {
        StringAttr tppMicroKernelName = rewriter.getStringAttr("tpp.add");
        rewriter.updateRootInPlace(
            linalgOp, [&]() { linalgOp.library_callAttr(tppMicroKernelName); });
        return success();
      }
    }
    if (isTPPGemm(linalgOp)) {
      if (failed(makeSIMDFriendly(linalgOp, rewriter)))
        return failure();
      // StringAttr tppMicroKernelName = rewriter.getStringAttr("tpp.matmul");
      // rewriter.updateRootInPlace(
      //     linalgOp, [&]() { linalgOp.library_callAttr(tppMicroKernelName);
      //     });
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
