//===- GeneralizeNamedOps.cpp ------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"
#include "TPP/Transforms/Transforms.h"
#include "TPP/Transforms/Utils/TransformUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace tpp;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_GENERALIZENAMEDOPS
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

struct LinalgGeneralizationPattern
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  LinalgGeneralizationPattern(MLIRContext *context, ControlGeneralizationFn fun,
                              PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern<linalg::LinalgOp>(context, benefit),
        controlFn(std::move(fun)) {}

  /// `matchAndRewrite` implementation that returns the significant
  /// transformed pieces of IR.
  FailureOr<linalg::GenericOp>
  returningMatchAndRewrite(linalg::LinalgOp op,
                           PatternRewriter &rewriter) const {
    return linalg::generalizeNamedOp(rewriter, op);
  }

  LogicalResult matchAndRewrite(linalg::LinalgOp op,
                                PatternRewriter &rewriter) const override {
    if (controlFn && !controlFn(op))
      return failure();

    return returningMatchAndRewrite(op, rewriter);
  }

private:
  ControlGeneralizationFn controlFn;
};

struct GeneralizeNamedOps
    : tpp::impl::GeneralizeNamedOpsBase<GeneralizeNamedOps> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());

    ControlGeneralizationFn controlFn = [](linalg::LinalgOp op) -> bool {
      return !(isa<linalg::FillOp>(op));
    };

    tpp::populateGeneralizeNamedOpsPatterns(patterns, controlFn);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace

// TODO: Add control function to Linalg generalization patterns upstream.
void tpp::populateGeneralizeNamedOpsPatterns(
    RewritePatternSet &patterns, ControlGeneralizationFn controlFn) {
  patterns.add<LinalgGeneralizationPattern>(patterns.getContext(), controlFn);
}
