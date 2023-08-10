//===- LinalgDeGeneralize.cpp ------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Tpp/TppOps.h"
#include "TPP/Dialect/Tpp/TppUtils.h"
#include "TPP/IR/StructuredOpMatcher.h"
#include "TPP/Passes.h"
#include "TPP/TransformUtils.h"
#include "TPP/Transforms.h"
#include "TPP/VNNIUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

struct LinalgDeGeneralize : LinalgDeGeneralizeBase<LinalgDeGeneralize> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(&getContext());
    linalg::populateLinalgDeGeneralizationPatterns(patterns);
    tpp::populateTppDeGeneralizationPatterns(patterns);
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
    using namespace tpp::structured_match;
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
    SmallVector<Value> inputOperands = linalgOp.getDpsInputOperands();
    SmallVector<Value> outputOperands = linalgOp.getDpsInitOperands();
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
    using namespace tpp::structured_match;
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
    SmallVector<Value> inputOperands = linalgOp.getDpsInputOperands();
    SmallVector<Value> outputOperands = linalgOp.getDpsInitOperands();
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
    using namespace tpp::structured_match;
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
    SmallVector<Value> inputOperands = linalgOp.getDpsInputOperands();
    SmallVector<Value> outputOperands = linalgOp.getDpsInitOperands();
    rewriter.replaceOpWithNewOp<linalg::FillOp>(
        linalgOp, linalgOp.getResultTypes(), inputOperands, outputOperands);
    return success();
  }
};

// From linalg.generic to TPP brgemm (VNNI).
struct TppBrgemmDeGeneralizationPattern
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  bool isTppVnniOp(linalg::GenericOp linalgOp) const {
    using MapList = ArrayRef<ArrayRef<AffineExpr>>;
    auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
    AffineExpr r1, p4, p5, r2, r3;
    bindDims(linalgOp.getContext(), r1, r2, p4, p5, r3);
    auto blockingFactor = vnni::utils::getVnniBlockingFactor(
        linalgOp->getOperands()[0].getType());
    if (!blockingFactor)
      return false;
    SmallVector<AffineMap> mapList;
    mapList = infer(
        {{r1, p4, r3}, {r1, r3.floorDiv(*blockingFactor), p5, r2}, {p4, p5}});

    using namespace tpp::structured_match;
    auto matmulMatcher =
        StructuredOpMatcher::make<linalg::GenericOp>()
            .operation(NumDpsInits(EqualsTo(1)))
            .operation(NumDpsInputs(EqualsTo(2)))
            .operation(NumRegions(EqualsTo(1)))
            .dim(MatchAll(), {mlir::utils::IteratorType::reduction,
                              mlir::utils::IteratorType::parallel,
                              mlir::utils::IteratorType::parallel,
                              mlir::utils::IteratorType::reduction,
                              mlir::utils::IteratorType::reduction})
            .input(MatchOne(0), HasMap(EqualsTo(mapList[0])))
            .input(MatchOne(1), HasMap(EqualsTo(mapList[1])))
            .output(MatchOne(0), HasMap(EqualsTo(mapList[2])))
            .region(MatchOne(0), WithOpChain<arith::MulFOp, arith::AddFOp>(
                                     /*captures=*/nullptr));
    return matmulMatcher.match(linalgOp);
  }

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!isTppVnniOp(linalgOp))
      return failure();
    SmallVector<Value> operands = linalgOp.getDpsInputOperands();
    SmallVector<Value> initOperands = linalgOp.getDpsInitOperands();
    operands.append(initOperands.begin(), initOperands.end());
    rewriter.replaceOpWithNewOp<tpp::BrgemmOp>(linalgOp, operands,
                                               operands.back().getType());
    return success();
  }
};

} // namespace

void mlir::linalg::populateLinalgDeGeneralizationPatterns(
    RewritePatternSet &patterns) {
  patterns.add<FillOpDeGeneralizationPattern, MatmulOpDeGeneralizationPattern,
               BatchReduceOpDeGeneralizationPattern>(patterns.getContext());
}

void mlir::tpp::populateTppDeGeneralizationPatterns(
    RewritePatternSet &patterns) {
  patterns.add<TppBrgemmDeGeneralizationPattern>(patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::linalg::createLinalgDeGeneralizationPass() {
  return std::make_unique<LinalgDeGeneralize>();
}
