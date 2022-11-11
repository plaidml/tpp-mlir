//===- MapLinalgToTpp.cpp ----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Tpp/TppOps.h"
#include "TPP/Dialect/Tpp/TppUtils.h"
#include "TPP/Passes.h"
#include "TPP/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

#define DEBUG_TYPE "linalg-map-to-tpp"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

// Return true if: 1) the region has a single block. 2) The block has two
// operations only (linalg.YieldOp and OP). 3) The operation result types are
// int or float.
// TODO: For now we assume the region to have only two operations: The YieldOp
// and the 'OP', meaning that the entire linalg.generic will map to a single
// tpp operation. If we do element-wise fusion at the linalg level this
// assumption does not hold anymore as now a linalg.generic can map to n tpp
// operations. If we support 1:n matching what should we do if the entire
// linalg.op cannot be replace by tpp operations?
template <typename OP> static bool hasOnlyScalarElementwiseOp(Region &region) {
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
static bool isTPPGemm(linalg::GenericOp linalgOp) {
  // structural and access pattern.
  SmallVector<utils::IteratorType> iteratorTypes =
      linalgOp.getIteratorTypesArray();
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

// Return true if the operation as 1 input and 1 output.
static bool hasOneInputOneOutput(linalg::GenericOp linalgOp) {
  return ((linalgOp.getNumDpsInputs() == 1) &&
          (linalgOp.getNumDpsInits() == 1));
}

static FailureOr<linalg::GenericOp>
mapLinalgToTppImpl(RewriterBase &rewriter, linalg::GenericOp linalgOp) {
  if (!hasStaticShape(linalgOp))
    return rewriter.notifyMatchFailure(linalgOp, "shape is not static");

  if (linalgOp.getLibraryCallAttr())
    return rewriter.notifyMatchFailure(linalgOp,
                                       "library_call attr already set");

  if (isTPPGemm(linalgOp)) {
    StringAttr tppMicroKernelName = rewriter.getStringAttr("tpp.matmul");
    rewriter.updateRootInPlace(
        linalgOp, [&]() { linalgOp.setLibraryCallAttr(tppMicroKernelName); });
    return linalgOp;
  }

  if (!linalg::isElementwise(linalgOp))
    return rewriter.notifyMatchFailure(linalgOp, "unmatched Linalg op");

  if (hasCopySemantics(linalgOp) && hasStaticShape(linalgOp)) {
    StringAttr tppMicroKernelName = rewriter.getStringAttr("tpp.identity");
    rewriter.updateRootInPlace(
        linalgOp, [&]() { linalgOp.setLibraryCallAttr(tppMicroKernelName); });
    return linalgOp;
  }

  // TODO: make sure we have a max(x, 0).
  if (hasOnlyScalarElementwiseOp<arith::MaxFOp>(linalgOp.getRegion()) &&
      hasStaticShape(linalgOp)) {
    StringAttr tppMicroKernelName = rewriter.getStringAttr("tpp.relu");
    rewriter.updateRootInPlace(
        linalgOp, [&]() { linalgOp.setLibraryCallAttr(tppMicroKernelName); });
    return linalgOp;
  }

  if (hasOnlyScalarElementwiseOp<arith::AddFOp>(linalgOp.getRegion()) &&
      hasStaticShape(linalgOp) && hasOneInputOneOutput(linalgOp)) {
    StringAttr tppMicroKernelName = rewriter.getStringAttr("tpp.add");
    rewriter.updateRootInPlace(
        linalgOp, [&]() { linalgOp.setLibraryCallAttr(tppMicroKernelName); });
    return linalgOp;
  }
  return rewriter.notifyMatchFailure(linalgOp, "unmatched linalg op");
}

namespace {

struct MapGenericOpToTpp : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    FailureOr<linalg::GenericOp> annotatedOp =
        mapLinalgToTppImpl(rewriter, linalgOp);
    if (failed(annotatedOp))
      return rewriter.notifyMatchFailure(linalgOp,
                                         "failed to map operation to tpp");
    return success();
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

FailureOr<linalg::GenericOp>
mlir::linalgx::mapLinalgToTpp(RewriterBase &rewriter,
                              linalg::GenericOp linalgOp) {
  return mapLinalgToTppImpl(rewriter, linalgOp);
}

void mlir::tpp::populateMapLinalgToTppPatterns(RewritePatternSet &patterns) {
  patterns.add<MapGenericOpToTpp>(patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createMapLinalgToTppPass() {
  return std::make_unique<MapToTpp>();
}
