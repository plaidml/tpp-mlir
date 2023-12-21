//===- LinalgConvertToTpp.cpp ------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Tpp/TppOps.h"
#include "TPP/IR/MatcherUtils.h"
#include "TPP/Passes.h"
#include "TPP/Transforms/Transforms.h"
#include "TPP/Transforms/Utils/TransformUtils.h"
#include "TPP/Transforms/Utils/ValueUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_CONVERTLINALGTOTPP
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

#define DEBUG_TYPE "linalg-convert-to-tpp"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

namespace {

// Convert a linalg.generic to a tpp operation.
struct ConvertGenericOpToTpp : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult rewriteToTppOp(linalg::GenericOp linalgOp,
                               PatternRewriter &rewriter) const {
    return rewriter.notifyMatchFailure(
        linalgOp, "failed to match to a known tpp operation");
  }

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!linalgOp.hasTensorSemantics()) {
      return rewriter.notifyMatchFailure(
          linalgOp, "Expect tensor type when mapping to tpp");
    }
    if (linalgOp.hasDynamicShape()) {
      return rewriter.notifyMatchFailure(
          linalgOp, "Expect static shape when mapping to tpp");
    }
    return rewriteToTppOp(linalgOp, rewriter);
  }
};

struct ConvertLinalgToTpp
    : public tpp::impl::ConvertLinalgToTppBase<ConvertLinalgToTpp> {
  void runOnOperation() override {
    MLIRContext *ctx = getOperation().getContext();
    RewritePatternSet patterns(ctx);
    tpp::populateConvertLinalgToTppPatterns(patterns);
    memref::SubViewOp::getCanonicalizationPatterns(patterns, ctx);
    linalg::populateLinalgDeGeneralizationPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // end namespace

void mlir::tpp::populateConvertLinalgToTppPatterns(
    RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<ConvertGenericOpToTpp>(patterns.getContext());
  // clang-format on
}
