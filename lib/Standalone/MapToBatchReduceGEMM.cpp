//===- MapToBatchReduceGEMM.cpp ----------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/Tpp/TppUtils.h"
#include "Standalone/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "Standalone/Passes.h.inc"

namespace {

struct DoItOnGeneric : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  // look for p p r p p r
  LogicalResult checkStructure(linalg::GenericOp linalgOp) const {
    ArrayAttr iteratorTypes = linalgOp.getIteratorTypes();
    if (iteratorTypes.size() != 6)
      return failure();
    if (!(isReductionIterator(iteratorTypes[0]) &&
          (isParallelIterator(iteratorTypes[1]) &&
           (isParallelIterator(iteratorTypes[2]) &&
            (isReductionIterator(iteratorTypes[3]) &&
             (isParallelIterator(iteratorTypes[4]) &&
              (isParallelIterator(iteratorTypes[5]))))))))
      return failure();
    return success();
  }

  LogicalResult checkAccessPatterns(linalg::GenericOp linalgOp) const {
    using MapList = ArrayRef<ArrayRef<AffineExpr>>;
    auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
    AffineExpr p1, p2, r1, p3, p4, r2;
    bindDims(linalgOp.getContext(), p1, p2, r1, p3, p4, r2);
    if (linalgOp.getIndexingMapsArray() !=
        infer({{p1, r1, p3, r2}, {p2, r1, r2, p4}, {p1, p2, p3, p4}}))
      return failure();
    return success();
  }

  LogicalResult checkBody(linalg::GenericOp linalgOp) const {
    if (tpp::hasMatmulBody(linalgOp))
      return success();
    return failure();
  }

  // Specific pattern (maybe too specific). Look for a blocked
  // matmul and map it to BRGEMM if the layout allows.
  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!tpp::hasStaticShape(linalgOp))
      return failure();
    if (failed(checkStructure(linalgOp)) ||
        failed(checkAccessPatterns(linalgOp)) || failed(checkBody(linalgOp)))
      return failure();

    return failure();
  }
};

struct MapToBatchReduceGEMM
    : public MapToBatchReduceGEMMBase<MapToBatchReduceGEMM> {
  void runOnOperation() override {
    RewritePatternSet patterns(getOperation().getContext());
    patterns.add<DoItOnGeneric>(patterns.getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // end namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createMapToBatchReduceGEMMPass() {
  return std::make_unique<MapToBatchReduceGEMM>();
}
