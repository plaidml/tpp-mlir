//===- InterchangeBlockConvToExposeMatmul.cpp --------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"
#include "TPP/TransformUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

// Interchange a blocked convolutions to expose a linalg.matmul.
struct InterchangeBlockConvToExposeMatmulImpl
    : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!linalgx::utils::isBlockedConvolution(genericOp))
      return failure();

    // clang-format off
    // N                [parallel]
    //  K               [parallel - blocked]
    //    P             [parallel]
    //      Q           [parallel]
    //        k         [parallel - block of K]
    //          C       [reduction - blocked]
    //            R     [reduction]
    //              S   [reduction]
    //                c [reduction - block of C]

    // expose matmul by interchange

    // N                [parallel]
    //  K               [parallel - blocked]
    //    P             [parallel]
    //      C           [reduction - blocked]
    //        R         [reduction]
    //          S       [reduction]
    //            Q     [parallel]
    //              k   [parallel - block of K]
    //                c [reduction - block of C]
    //
    // Matmul: m = %Q, n = %k and k = %c
    // clang-format on

    SmallVector<unsigned> interchangeVector = {0, 1, 2, 5, 6, 7, 3, 4, 8};
    FailureOr<linalg::GenericOp> maybeInterchange =
        interchangeGenericOp(rewriter, genericOp, interchangeVector);
    if (failed(maybeInterchange))
      return failure();
    return success();
  }
};

struct InterchangeBlockConvToExposeMatmul
    : public InterchangeBlockConvToExposeMatmulBase<
          InterchangeBlockConvToExposeMatmul> {
  void runOnOperation() override {
    RewritePatternSet patterns(getOperation().getContext());
    patterns.add<InterchangeBlockConvToExposeMatmulImpl>(patterns.getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // end namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createInterchangeBlockConvToExposeMatmulPass() {
  return std::make_unique<InterchangeBlockConvToExposeMatmul>();
}
