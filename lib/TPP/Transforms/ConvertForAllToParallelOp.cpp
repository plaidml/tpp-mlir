//===- ConvertForAllToParallelOp.cpp -----------------------------*- C++-*-===//
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

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_CONVERTFORALLTOPARALLELOP
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

struct ConvertForAllToParallelOpImpl : public OpRewritePattern<scf::ForallOp> {
  using OpRewritePattern<scf::ForallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForallOp forallOp,
                                PatternRewriter &rewriter) const override {
    // Bail-out if we are not at memref.
    if (forallOp->getNumResults() != 0)
      return failure();
    Location loc = forallOp.getLoc();
    SmallVector<Value> lowerBounds = getValueOrCreateConstantIndexOp(
        rewriter, loc, forallOp.getMixedLowerBound());
    SmallVector<Value> upperBounds = getValueOrCreateConstantIndexOp(
        rewriter, loc, forallOp.getMixedUpperBound());
    SmallVector<Value> steps =
        getValueOrCreateConstantIndexOp(rewriter, loc, forallOp.getMixedStep());
    rewriter.replaceOpWithNewOp<scf::ParallelOp>(
        forallOp, lowerBounds, upperBounds, steps,
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange regionArgs) {
          IRMapping mapping;
          mapping.map(forallOp.getInductionVars(), regionArgs);
          Block *forallOpBlock = forallOp.getBody();
          for (auto &nestedOp : forallOpBlock->without_terminator())
            nestedBuilder.clone(nestedOp, mapping);
        });
    return success();
  }
};

struct ConvertForAllToParallelOp
    : public tpp::impl::ConvertForAllToParallelOpBase<
          ConvertForAllToParallelOp> {
  void runOnOperation() override {
    RewritePatternSet patterns(getOperation().getContext());
    patterns.add<ConvertForAllToParallelOpImpl>(patterns.getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // end namespace
