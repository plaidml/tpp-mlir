//===- RaiseToParallelLoop.cpp -----------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

struct RaiseToParallelLoopWithAttribute : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    if (forOp.hasIterOperands() || !forOp->hasAttr("parallel"))
      return failure();

    auto bodyBuilder = [&](OpBuilder &builder, Location loc,
                           ValueRange iterVals, ValueRange) {
      Block &innerBody = forOp.getLoopBody().front();
      IRMapping mapping;
      mapping.map(innerBody.getArguments(),
                  iterVals.take_back(innerBody.getNumArguments()));
      for (Operation &op : innerBody.without_terminator())
        builder.clone(op, mapping);
    };

    rewriter.replaceOpWithNewOp<scf::ParallelOp>(
        forOp, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(),
        std::nullopt, bodyBuilder);
    return success();
  }
};

struct RaiseToParallelLoop
    : public RaiseToParallelLoopBase<RaiseToParallelLoop> {
  void runOnOperation() override {
    RewritePatternSet patterns(getOperation().getContext());
    patterns.add<RaiseToParallelLoopWithAttribute>(patterns.getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createRaiseToParallelLoopPass() {
  return std::make_unique<RaiseToParallelLoop>();
}
