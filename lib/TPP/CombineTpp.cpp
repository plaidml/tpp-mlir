//===- CombineTpp.cpp --------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Tpp/TppOps.h"
#include "TPP/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

struct CombineTppOpPattern : public OpRewritePattern<tpp::ReluOp> {
  using OpRewritePattern<tpp::ReluOp>::OpRewritePattern;

  // TODO this is not a sufficient check.
  static bool isVNNILayout(Operation *brgemmOp) {
    if (!brgemmOp->getOpOperands()[0]
             .get()
             .getType()
             .cast<ShapedType>()
             .getElementType()
             .isBF16())
      return false;
    return (brgemmOp->getOpOperands()[0]
                    .get()
                    .getType()
                    .cast<ShapedType>()
                    .getRank() +
                1 ==
            brgemmOp->getOpOperands()[1]
                .get()
                .getType()
                .cast<ShapedType>()
                .getRank());
  }

  LogicalResult matchAndRewrite(tpp::ReluOp reluOp,
                                PatternRewriter &rewriter) const override {

    auto definingOp = reluOp->getOperands()[0].getDefiningOp();
    Operation *addOp = NULL;
    for (Operation *user : definingOp->getUsers()) {
      if (user == definingOp || user == reluOp ||
          isa<memref::DeallocOp>(user)) {
        continue;
      }
      // assert(user.isa<tpp::AddBCastOp>());
      addOp = user;
      break;
    }

    if (addOp == NULL) {
      return failure();
    }
    auto brgemmResultBuffer = addOp->getOperands()[0].getDefiningOp();
    Operation *brgemmOp = NULL;
    for (Operation *user : brgemmResultBuffer->getUsers()) {
      if (isa<memref::CastOp>(user)) {
        for (Operation *castUser : user->getUsers()) {
          if (isa<tpp::BrgemmOp>(castUser) ||
              isa<tpp::VNNIBrgemmOp>(castUser)) {
            brgemmOp = castUser;
            break;
          }
        }
      }
      if (brgemmOp != NULL)
        break;
    }

    if (brgemmOp == NULL) {
      return failure();
    }
    Value result;
    // Replace brgemm-addbcast-relu ops into one large tpp op
    if (isVNNILayout(brgemmOp)) {
      result = rewriter
                   .create<tpp::FusedVNNIBrgemmOp>(
                       reluOp.getLoc(), brgemmOp->getOpOperands()[0].get(),
                       brgemmOp->getOpOperands()[1].get(),
                       addOp->getOpOperands()[1].get(),
                       reluOp->getOpOperands()[1].get())
                   ->getResult(0);
    } else {
      result = rewriter
                   .create<tpp::FusedBrgemmOp>(
                       reluOp.getLoc(), brgemmOp->getOpOperands()[0].get(),
                       brgemmOp->getOpOperands()[1].get(),
                       addOp->getOpOperands()[1].get(),
                       reluOp->getOpOperands()[1].get())
                   ->getResult(0);
    }
    rewriter.eraseOp(reluOp);
    rewriter.eraseOp(addOp);
    rewriter.eraseOp(brgemmOp);
    return success();
  }
};

void populatePatterns(RewritePatternSet &patterns) {
  patterns.add<CombineTppOpPattern>(patterns.getContext());
}

struct CombineTppOps : public CombineTppOpsBase<CombineTppOps> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populatePatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::tpp::createCombineTppPass() {
  return std::make_unique<CombineTppOps>();
}
