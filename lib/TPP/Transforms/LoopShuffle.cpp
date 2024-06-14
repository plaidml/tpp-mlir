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
#include "mlir/Transforms/RegionUtils.h"
#include <list>
#include <map>

namespace mlir {
namespace tpp {
#define GEN_PASS_DECL_LOOPSHUFFLEPASS
#define GEN_PASS_DEF_LOOPSHUFFLEPASS
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

using namespace mlir;
using namespace mlir::scf;
using namespace std;

namespace mlir {
namespace tpp {

struct LoopShuffle : OpRewritePattern<scf::ForallOp> {
  using OpRewritePattern<scf::ForallOp>::OpRewritePattern;

  LoopShuffle(MLIRContext *ctx, LoopShufflePassOptions &options)
      : OpRewritePattern(ctx), options(options) {}

  LogicalResult matchAndRewrite(scf::ForallOp op,
                                PatternRewriter &rewriter) const override {

    if (options.shuffleOrder.size() != op.getInductionVars().size())
      return failure();
    for (size_t i = 0; i < op.getInductionVars().size(); i++) {
      bool match = false;
      for (size_t j = 0; j < options.shuffleOrder.size(); j++)
        if (i == options.shuffleOrder[j]) {
          match = true;
          break;
        }
      if (!match) {
        return failure();
      }
    }
    xsmm::BrgemmOp brgemmOp = NULL;
    static list<scf::ForallOp> visitedForallOp;
    if (std::find(visitedForallOp.begin(), visitedForallOp.end(), op) !=
        visitedForallOp.end())
      return failure();
    for (auto oper = op.getBody()->getOperations().begin();
         oper != op.getBody()->getOperations().end(); oper++)
      if (dyn_cast<xsmm::BrgemmOp>(oper)) {
        brgemmOp = dyn_cast<xsmm::BrgemmOp>(oper);
        break;
      }
    if (brgemmOp == NULL)
      return failure();
    SmallVector<int64_t> lbs, ubs, steps;

    for (size_t i = 0; i < op.getStaticLowerBound().size(); i++) {
      lbs.push_back(op.getStaticLowerBound()[options.shuffleOrder[i]]);
    }
    for (size_t i = 0; i < op.getStaticUpperBound().size(); i++) {
      ubs.push_back(op.getStaticUpperBound()[options.shuffleOrder[i]]);
    }
    for (size_t i = 0; i < op.getStaticStep().size(); i++) {
      steps.push_back(op.getStaticStep()[options.shuffleOrder[i]]);
    }

    op.setStaticLowerBound(lbs);
    op.setStaticUpperBound(ubs);
    op.setStaticStep(steps);

    SmallVector<Value> tempValueMap(op.getInductionVars().size());
    SmallVector<int64_t> tempIndexMap(op.getInductionVars().size());
    for (size_t i = 0; i < op.getInductionVars().size(); i++) {
      for (size_t j = 0; j < options.shuffleOrder.size(); j++) {
        if (i == options.shuffleOrder[j]) {
          auto tempValue =
              rewriter.create<arith::ConstantIndexOp>(op.getLoc(), j);
          replaceAllUsesInRegionWith(op.getInductionVar(i), tempValue,
                                     op.getRegion());
          tempValueMap[i] = tempValue;
          tempIndexMap[i] = j;
          break;
        }
      }
    }
    for (size_t i = 0; i < op.getInductionVars().size(); i++) {
      replaceAllUsesInRegionWith(
          tempValueMap[i], op.getInductionVar(tempIndexMap[i]), op.getRegion());
    }
    visitedForallOp.push_back(op);
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
