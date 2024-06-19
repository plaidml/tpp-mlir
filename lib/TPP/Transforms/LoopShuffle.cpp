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
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include <iostream>
#include <list>

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

static LogicalResult loopShuffle(scf::ForallOp op,
                                 ArrayRef<unsigned> shuffleOrder) {
  OpBuilder b(op);
  IRRewriter rewriter(b.getContext());
  if (shuffleOrder.size() != op.getInductionVars().size())
    return rewriter.notifyMatchFailure(op, "Number of indices incorrect");
  for (size_t i = 0; i < op.getInductionVars().size(); i++) {
    bool match = false;
    for (size_t j = 0; j < shuffleOrder.size(); j++)
      if (i == shuffleOrder[j]) {
        match = true;
        break;
      }
    if (!match) {
      return rewriter.notifyMatchFailure(op, "Indices missed");
    }
  }

  SmallVector<int64_t> lbs, ubs, steps;

  for (size_t i = 0; i < op.getStaticLowerBound().size(); i++) {
    lbs.push_back(op.getStaticLowerBound()[shuffleOrder[i]]);
  }
  for (size_t i = 0; i < op.getStaticUpperBound().size(); i++) {
    ubs.push_back(op.getStaticUpperBound()[shuffleOrder[i]]);
  }
  for (size_t i = 0; i < op.getStaticStep().size(); i++) {
    steps.push_back(op.getStaticStep()[shuffleOrder[i]]);
  }

  op.setStaticLowerBound(lbs);
  op.setStaticUpperBound(ubs);
  op.setStaticStep(steps);

  SmallVector<Value> tempValueMap(op.getInductionVars().size());
  SmallVector<int64_t> tempIndexMap(op.getInductionVars().size());
  for (size_t i = 0; i < op.getInductionVars().size(); i++) {
    for (size_t j = 0; j < shuffleOrder.size(); j++) {
      if (i == shuffleOrder[j]) {
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
    rewriter.eraseOp(tempValueMap[i].getDefiningOp());
  }
  return success();
}

struct LoopShufflePass : public impl::LoopShufflePassBase<LoopShufflePass> {

  using LoopShufflePassBase::LoopShufflePassBase;

  void runOnOperation() override {
    scf::ForallOp failedForallOp;
    auto walkResult = getOperation()->walk([&](scf::ForallOp forallOp) {
      if (failed(loopShuffle(forallOp, shuffleOrder))) {
        failedForallOp = forallOp;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      failedForallOp->emitWarning("Failed to shuffle the loop");
  }
};

} // namespace tpp
} // namespace mlir
