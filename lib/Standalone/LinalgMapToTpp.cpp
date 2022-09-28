//===- LinalgMapToTpp.cpp ----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/Mathx/MathxOps.h"
#include "Standalone/Dialect/Tpp/TppOps.h"
#include "Standalone/Dialect/Tpp/TppUtils.h"
#include "Standalone/Passes.h"
#include "Standalone/Transforms.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "Standalone/Passes.h.inc"

#define DEBUG_TYPE "linalg-map-to-tpp"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

namespace {

struct MapGenericOpToTpp : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  // Return true if: 1) the region has a single block. 2) The block has two
  // operations only (linalg.YieldOp and OP). 3) The operation result types are
  // int or float.
  // TODO: For now we assume the region to have only two operations: The YieldOp
  // and the 'OP', meaning that the entire linalg.generic will map to a single
  // tpp operation. If we do element-wise fusion at the linalg level this
  // assumption does not hold anymore as now a linalg.generic can map to n tpp
  // operations. If we support 1:n matching what should we do if the entire
  // linalg.op cannot be replace by tpp operations?
  template <typename OP> bool hasOnlyScalarElementwiseOp(Region &region) const {
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

  // Return true if the linalg.generic maps to a tpp.gemm.
  bool isTPPGemm(linalg::GenericOp linalgOp) const {
    // structural and access pattern.
    ArrayAttr iteratorTypes = linalgOp.getIteratorTypes();
    if (iteratorTypes.size() != 3)
      return false;
    if (!(linalg::isParallelIterator(iteratorTypes[0]) &&
          linalg::isParallelIterator(iteratorTypes[1]) &&
          linalg::isReductionIterator(iteratorTypes[2])))
      return false;
    using MapList = ArrayRef<ArrayRef<AffineExpr>>;
    auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
    AffineExpr i, j, k;
    bindDims(linalgOp.getContext(), i, j, k);
    if (linalgOp.getIndexingMapsArray() != infer({{i, k}, {k, j}, {i, j}}))
      return false;
    // operations and operands.
    return hasMatmulBody(linalgOp);
  }

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!hasStaticShape(linalgOp))
      return rewriter.notifyMatchFailure(linalgOp, "shape is not static");

    if (linalgOp.library_callAttr())
      return rewriter.notifyMatchFailure(linalgOp,
                                         "library_call attr already set");

    if (isTPPGemm(linalgOp)) {
      StringAttr tppMicroKernelName = rewriter.getStringAttr("tpp.matmul");
      rewriter.updateRootInPlace(
          linalgOp, [&]() { linalgOp.library_callAttr(tppMicroKernelName); });
      return success();
    }

    if (!linalg::isElementwise(linalgOp))
      return rewriter.notifyMatchFailure(linalgOp, "unmatched Linalg op");

    if (hasCopySemantics(linalgOp) && hasStaticShape(linalgOp)) {
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

    return rewriter.notifyMatchFailure(linalgOp, "unmatched Linalg op");
  }
};

struct MapToTpp : public LinalgMapToTppBase<MapToTpp> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateMapLinalgToTppPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // end namespace

void mlir::tpp::populateMapLinalgToTppPatterns(RewritePatternSet &patterns) {
  patterns.add<MapGenericOpToTpp>(patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createMapLinalgToTppPass() {
  return std::make_unique<MapToTpp>();
}
