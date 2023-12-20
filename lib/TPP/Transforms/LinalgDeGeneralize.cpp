//===- LinalgDeGeneralize.cpp ------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Tpp/TppOps.h"
#include "TPP/IR/StructuredOpMatcher.h"
#include "TPP/Passes.h"
#include "TPP/Transforms/Transforms.h"
#include "TPP/Transforms/Utils/TransformUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_LINALGDEGENERALIZE
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

struct LinalgDeGeneralize
    : tpp::impl::LinalgDeGeneralizeBase<LinalgDeGeneralize> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(&getContext());
    linalg::populateLinalgDeGeneralizationPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(func.getBody(), std::move(patterns));
  }
};

// From linalg.generic to linalg.matmul.
struct MatmulOpDeGeneralizationPattern
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  bool isMatmulOp(linalg::LinalgOp linalgOp) const {
    if (isa_and_nonnull<linalg::MatmulOp>(linalgOp))
      return true;
    using namespace mlir::structured_match;
    using MapList = ArrayRef<ArrayRef<AffineExpr>>;
    auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
    AffineExpr i, j, k;
    bindDims(linalgOp->getContext(), i, j, k);
    auto mapList = infer({{i, k}, {k, j}, {i, j}});
    auto matmulMatcher =
        StructuredOpMatcher::make<linalg::GenericOp>()
            .operation(NumDpsInits(EqualsTo(1)))
            .operation(NumDpsInputs(EqualsTo(2)))
            .operation(NumRegions(EqualsTo(1)))
            .dim(MatchAll(), {mlir::utils::IteratorType::reduction,
                              mlir::utils::IteratorType::parallel,
                              mlir::utils::IteratorType::parallel})
            .input(MatchOne(0), HasMap(EqualsTo(mapList[0])))
            .input(MatchOne(1), HasMap(EqualsTo(mapList[1])))
            .output(MatchOne(0), HasMap(EqualsTo(mapList[2])))
            .region(MatchOne(0), WithOpChain<arith::MulFOp, arith::AddFOp>(
                                     /*captures=*/nullptr));
    return matmulMatcher.match(linalgOp);
  }

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!isMatmulOp(linalgOp))
      return failure();
    SmallVector<Value> inputOperands = linalgOp.getDpsInputs();
    SmallVector<Value> outputOperands = linalgOp.getDpsInits();
    rewriter.replaceOpWithNewOp<linalg::MatmulOp>(
        linalgOp, linalgOp.getResultTypes(), inputOperands, outputOperands);
    return success();
  }
};

// From linalg.generic to linalg.batch_reduce_matmul.
struct BatchReduceOpDeGeneralizationPattern
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  bool isBrgemmOp(linalg::LinalgOp linalgOp) const {
    if (isa_and_nonnull<linalg::BatchReduceMatmulOp>(linalgOp))
      return true;
    using namespace mlir::structured_match;
    using MapList = ArrayRef<ArrayRef<AffineExpr>>;
    auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
    AffineExpr r, i, j, k;
    bindDims(linalgOp->getContext(), r, i, j, k);
    auto mapList = infer({{r, i, k}, {r, k, j}, {i, j}});
    auto brgemmMatcher =
        StructuredOpMatcher::make<linalg::GenericOp>()
            .operation(NumDpsInits(EqualsTo(1)))
            .operation(NumDpsInputs(EqualsTo(2)))
            .operation(NumRegions(EqualsTo(1)))
            .dim(MatchAll(), {mlir::utils::IteratorType::reduction,
                              mlir::utils::IteratorType::parallel,
                              mlir::utils::IteratorType::parallel,
                              mlir::utils::IteratorType::reduction})
            .input(MatchOne(0), HasMap(EqualsTo(mapList[0])))
            .input(MatchOne(1), HasMap(EqualsTo(mapList[1])))
            .output(MatchOne(0), HasMap(EqualsTo(mapList[2])))
            .region(MatchOne(0), WithOpChain<arith::MulFOp, arith::AddFOp>(
                                     /*captures=*/nullptr));
    return brgemmMatcher.match(linalgOp);
  }

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!isBrgemmOp(linalgOp))
      return failure();
    SmallVector<Value> inputOperands = linalgOp.getDpsInputs();
    SmallVector<Value> outputOperands = linalgOp.getDpsInits();
    rewriter.replaceOpWithNewOp<linalg::BatchReduceMatmulOp>(
        linalgOp, linalgOp.getResultTypes(), inputOperands, outputOperands);
    return success();
  }
};

// From linalg.generic to linalg.fillOp.
struct FillOpDeGeneralizationPattern
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  bool isFillOp(linalg::LinalgOp linalgOp) const {
    if (isa_and_nonnull<linalg::FillOp>(linalgOp))
      return true;
    using namespace mlir::structured_match;
    auto fillMatcher =
        StructuredOpMatcher::make<linalg::GenericOp>()
            .operation(NumDpsInits(EqualsTo(1)))
            .operation(NumDpsInputs(EqualsTo(1)))
            .operation(NumRegions(EqualsTo(1)))
            .dim(MatchAll(), mlir::utils::IteratorType::parallel)
            .output(MatchAll(), HasMap(Identity()))
            .input(MatchAll(), HasMap(ProjectedPermutation()))
            .input(MatchAll(), HasRank({HasRank::SCALAR}))
            .region(MatchOne(0),
                    WithSingleOp<linalg::YieldOp>(/*captures=*/nullptr));
    return fillMatcher.match(linalgOp);
  }

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!isFillOp(linalgOp))
      return failure();
    SmallVector<Value> inputOperands = linalgOp.getDpsInputs();
    SmallVector<Value> outputOperands = linalgOp.getDpsInits();
    rewriter.replaceOpWithNewOp<linalg::FillOp>(
        linalgOp, linalgOp.getResultTypes(), inputOperands, outputOperands);
    return success();
  }
};

} // namespace

void mlir::linalg::populateLinalgDeGeneralizationPatterns(
    RewritePatternSet &patterns) {
  patterns.add<FillOpDeGeneralizationPattern, MatmulOpDeGeneralizationPattern,
               BatchReduceOpDeGeneralizationPattern>(patterns.getContext());
}
