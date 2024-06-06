//===- LoopShuffle.cpp -----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file shuffles parallel loop based on user input
//
//===----------------------------------------------------------------------===//
#include "TPP/Dialect/Xsmm/XsmmOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace tpp {
#define GEN_PASS_DECL_LOOPSHUFFLEPASS
#define GEN_PASS_DEF_LOOPSHUFFLEPASS
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

using namespace mlir;
using namespace mlir::scf;

namespace mlir {
namespace tpp {

struct LoopShuffle : OpRewritePattern<scf::ForallOp> {
  using OpRewritePattern<scf::ForallOp>::OpRewritePattern;

  LoopShuffle(MLIRContext *ctx, LoopShufflePassOptions &options)
      : OpRewritePattern(ctx), options(options) {}

  LogicalResult matchAndRewrite(scf::ForallOp op,
                                PatternRewriter &rewriter) const override {
    xsmm::BrgemmOp brgemmOp = NULL;
    for (auto oper = op.getBody()->getOperations().begin();
         oper != op.getBody()->getOperations().end(); oper++)
      if (dyn_cast<xsmm::BrgemmOp>(oper)) {
        brgemmOp = dyn_cast<xsmm::BrgemmOp>(oper);
        break;
      }
    if (brgemmOp == NULL)
      return failure();

    for (size_t i = 0; i < op.getInductionVars().size(); i++) {
      auto sourceArg = op.getInductionVars()[i];
      auto replacementArg = op.getInductionVars()[options.shuffleOrder[i]];
      sourceArg.replaceAllUsesWith(replacementArg);
    }
    op.erase();
    return success();
  }

private:
  LoopShufflePassOptions options;
};

struct LoopShufflePass : public impl::LoopShufflePassBase<LoopShufflePass> {

  LoopShufflePass() {}

  LoopShufflePass(const LoopShufflePassOptions &options) {
    this->shuffleOrder = options.shuffleOrder;
  }

  void populateCombinePatterns(RewritePatternSet &patterns,
                               LoopShufflePassOptions options) {
    patterns.add<LoopShuffle>(patterns.getContext(), options);
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateCombinePatterns(patterns,
                            LoopShufflePassOptions{this->shuffleOrder});
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};
} // namespace tpp
} // namespace mlir
