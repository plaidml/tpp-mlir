//===- LoopExpansion.cpp -----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file  splits parallel loop into scf fors.
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
#define GEN_PASS_DECL_LOOPEXPANSIONPASS
#define GEN_PASS_DEF_LOOPEXPANSIONPASS
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

using namespace mlir;
using namespace mlir::scf;

namespace mlir {
namespace tpp {

struct LoopExpansion : OpRewritePattern<scf::ForallOp> {
  using OpRewritePattern<scf::ForallOp>::OpRewritePattern;

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
    auto ub = op.getStaticUpperBound().begin();
    auto step = op.getStaticStep().begin();
    rewriter.setInsertionPoint(op->getParentOp());
    for (auto lb = op.getStaticLowerBound().begin();
         lb != op.getStaticLowerBound().end() &&
         ub != op.getStaticUpperBound().end() &&
         step != op.getStaticStep().end();
         lb++, ub++, step++) {
      auto lowerBound =
          rewriter.create<arith::ConstantIndexOp>(op.getLoc(), *lb);
      auto upperBound =
          rewriter.create<arith::ConstantIndexOp>(op.getLoc(), *ub);
      auto stepVal =
          rewriter.create<arith::ConstantIndexOp>(op.getLoc(), *step);
      auto forOp = rewriter.create<scf::ForOp>(op.getLoc(), lowerBound,
                                               upperBound, stepVal);
      rewriter.setInsertionPoint(&forOp.getBody()->front());
    }
    IRMapping mapping;
    rewriter.clone(*op.getOperation(), mapping);
    op.erase();
    return success();
  }
};

struct LoopExpansionPass
    : public impl::LoopExpansionPassBase<LoopExpansionPass> {

  void populateCombinePatterns(RewritePatternSet &patterns) {
    patterns.add<LoopExpansion>(patterns.getContext());
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateCombinePatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};
} // namespace tpp
} // namespace mlir
