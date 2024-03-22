//===-ConvertAddInplacePass.cpp ----------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file converts add to an in-place add operation
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_CONVERTADDINPLACEPASS
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

using namespace mlir;
using namespace mlir::linalg;

namespace mlir {
namespace tpp {

struct ConvertAddInplace : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {

    if (op.getBody()->getOperations().size() != 2)
      return failure();
    auto addf = dyn_cast<arith::AddFOp>(&op.getBody()->getOperations().front());
    if (!addf)
      return failure();
    if (op.getNumOperands() == 2)
      return failure();
    if (op.getInputs()[0] == op.getInputs()[1])
      return failure();
    SmallVector<AffineMap> indexingMaps;
    SmallVector<utils::IteratorType> iteratorTypes;
    for (auto iteratorTypesArray : op.getIteratorTypesArray()) {
      iteratorTypes.push_back(iteratorTypesArray);
    }

    Value inputs, outputs;
    if (op.getIndexingMapsArray()[1] ==
        rewriter.getMultiDimIdentityMap(
            op.getIndexingMapsArray()[1].getNumDims())) {
      indexingMaps.push_back(op.getIndexingMapsArray()[0]);
      indexingMaps.push_back(op.getIndexingMapsArray()[1]);
      inputs = op.getInputs()[0];
      outputs = op.getInputs()[1];
    } else {
      indexingMaps.push_back(op.getIndexingMapsArray()[1]);
      indexingMaps.push_back(op.getIndexingMapsArray()[0]);
      inputs = op.getInputs()[1];
      outputs = op.getInputs()[0];
    }
    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, op.getResultTypes(), inputs, outputs, indexingMaps, iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
          auto scalarOp = builder.create<arith::AddFOp>(loc, regionArgs);
          builder.create<linalg::YieldOp>(loc, scalarOp.getResult());
        });
    return success();
  }
};

struct ConvertAddInplacePass
    : public impl::ConvertAddInplacePassBase<ConvertAddInplacePass> {
  void populateCombinePatterns(RewritePatternSet &patterns) {
    patterns.add<ConvertAddInplace>(patterns.getContext());
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateCombinePatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};
} // namespace tpp
} // namespace mlir
