//===- FoldAddIntoDest.cpp ---------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"
#include "TPP/Transforms/Utils/ValueUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
using namespace mlir;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_FOLDADDINTODEST
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

/// Replace a linalg.add where its linalg-contraction operand - with a
/// zero-filled destination - is dominated by the `other` linalg operand,
/// by passing `other` as the destination of the contraction.
struct FoldAddIntoDestRewrite : public OpRewritePattern<linalg::AddOp> {
  using OpRewritePattern<linalg::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::AddOp addOp,
                                PatternRewriter &rewriter) const override {
    Value dominatingOperand = nullptr;
    linalg::LinalgOp dominatedOp = nullptr;
    {
      auto firstOperand = addOp.getOperand(0);
      auto secondOperand = addOp.getOperand(1);

      // Can only put one of addOp's operands in the dest/out arg of the other's
      // defining op based on suitable dominance.
      if (auto secondOp = secondOperand.getDefiningOp<linalg::LinalgOp>()) {
        DominanceInfo domInfo(secondOp);
        if (domInfo.properlyDominates(firstOperand, secondOp)) {
          dominatingOperand = firstOperand;
          dominatedOp = secondOp;
        }
      }
      if (auto firstOp = firstOperand.getDefiningOp<linalg::LinalgOp>()) {
        DominanceInfo domInfo(firstOp);
        if (domInfo.properlyDominates(secondOperand, firstOp)) {
          dominatingOperand = secondOperand;
          dominatedOp = firstOp;
        }
      }
      if (!dominatingOperand || !dominatedOp)
        return failure();
      // NB: As linalg.add's generalisation ignores the out argument in its
      //     region there is no need to perform checks on addOp's out argument.
    }

    // When the dominated op has a single-result and is a contraction, the op
    // accumulates on the out argument, starting from the supplied out argument.
    // E.g., AddOp is not a contraction and hence ignores its out arg's value.
    auto dominatedDestOp =
        dyn_cast<DestinationStyleOpInterface>((Operation *)dominatedOp);
    if (dominatedOp->getNumResults() != 1 ||
        !linalg::isaContractionOpInterface(dominatedOp) ||
        (!dominatedDestOp || dominatedDestOp.getNumDpsInits() != 1))
      return rewriter.notifyMatchFailure(
          dominatedOp, "expected dominated op to be single-result "
                       "destination-passing contraction");

    // As the dominated op was already accumulating on its out argument, it is
    // only safe to discard its current out arg when it is the additive zero.
    auto *dominatedDest =
        dominatedDestOp.getDpsInitOperand(0)->get().getDefiningOp();
    if (!mlir::utils::isZeroOp(dominatedDest))
      return rewriter.notifyMatchFailure(
          dominatedOp, "expected dominated op's dest to be additive zero");
    // TODO: If the other op is a contraction and has additive zero as dest, we
    // can swap the dests and achieve the proper sum, given suitable dominance.

    // Replace the additive-zero out argument of the dominated op by the
    // dominating summand. This makes the dominated op's result the sum of both
    // of addOp's arguments - therefore we replace addOp and it uses by it.
    rewriter.modifyOpInPlace(
        dominatedOp, [&]() { dominatedOp->setOperand(2, dominatingOperand); });
    rewriter.replaceAllOpUsesWith(addOp, dominatedOp->getResult(0));
    return success();
  }
};

/// Replace linalg.add when destination passing suffices for achieving the sum.
struct FoldAddIntoDest
    : public tpp::impl::FoldAddIntoDestBase<FoldAddIntoDest> {

  void runOnOperation() override {
    auto *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<FoldAddIntoDestRewrite>(ctx);

    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace
