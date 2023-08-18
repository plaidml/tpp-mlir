//===- RewriteBatchMatmulToMatmul.cpp ----------------------------*- C++-*-===//
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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

struct RankReducedExtractSliceOp
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    // Limit the replacement to sliceOp with batch matmul as users.
    if (!llvm::all_of(sliceOp->getUsers(), [](Operation *user) {
          return isa<linalg::BatchMatmulOp>(user);
        })) {
      return failure();
    }
    RankedTensorType resultType = sliceOp.getType();
    SmallVector<OpFoldResult> offsets = sliceOp.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = sliceOp.getMixedSizes();
    SmallVector<OpFoldResult> strides = sliceOp.getMixedStrides();
    auto reassociation = linalg::getReassociationMapForFoldingUnitDims(sizes);
    if (!reassociation ||
        reassociation->size() == static_cast<size_t>(resultType.getRank())) {
      return failure();
    }
    auto rankReducedType =
        tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
            reassociation->size(), sliceOp.getSourceType(), offsets, sizes,
            strides)
            .cast<RankedTensorType>();

    Location loc = sliceOp.getLoc();
    Value newSlice = rewriter.create<tensor::ExtractSliceOp>(
        loc, rankReducedType, sliceOp.getSource(), offsets, sizes, strides);
    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        sliceOp, resultType, newSlice, *reassociation);
    return success();
  }
};

struct RewriteBatchMatmulToMatmulImpl
    : public OpRewritePattern<linalg::BatchMatmulOp> {
  using OpRewritePattern<linalg::BatchMatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::BatchMatmulOp linalgOp,
                                PatternRewriter &rewriter) const override {
    auto operands = linalgOp.getDpsInputOperands();
    operands.append(linalgOp.getDpsInitOperands());
    SmallVector<Value> matmulOperands;
    for (OpOperand *operand : operands) {
      tensor::ExpandShapeOp expandOp =
          operand->get().getDefiningOp<tensor::ExpandShapeOp>();
      if (!expandOp || expandOp.getSrcType().getRank() != 2)
        return failure();
      matmulOperands.push_back(expandOp.getSrc());
    }
    Value outputOperand = matmulOperands.back();
    matmulOperands.pop_back();
    rewriter.replaceOpWithNewOp<linalg::MatmulOp>(
        linalgOp, outputOperand.getType(), matmulOperands, outputOperand);
    return success();
  }
};

struct RewriteBatchMatmulToMatmul
    : public RewriteBatchMatmulToMatmulBase<RewriteBatchMatmulToMatmul> {
  void runOnOperation() override {
    auto &ctx = getContext();
    IRRewriter rewriter(&ctx);
    // Step 1. tiling.
    getOperation()->walk([&](linalg::BatchMatmulOp batchMatmulOp) {
      if (batchMatmulOp.hasBufferSemantics())
        return signalPassFailure();
      SmallVector<OpFoldResult> tiles(
          batchMatmulOp.getNumLoops(),
          getAsIndexOpFoldResult(rewriter.getContext(), 0));
      tiles[0] = getAsIndexOpFoldResult(rewriter.getContext(), 1);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(batchMatmulOp);
      auto tilingResult = linalg::tileToForallOpUsingTileSizes(
          rewriter, cast<TilingInterface>(batchMatmulOp.getOperation()), tiles,
          /*mapping=*/std::nullopt);
      if (failed(tilingResult))
        return signalPassFailure();
      rewriter.replaceOp(batchMatmulOp, tilingResult->tileOp->getResults());
    });

    // Step2:
    // - replace extract/insert slice with ranked reduced extract/insert slice
    // and expand shape ops.
    // - replace linalg.batch_matmul with linalg.matmul.
    RewritePatternSet patterns(&ctx);
    linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
    tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
    patterns.add<RankReducedExtractSliceOp, RewriteBatchMatmulToMatmulImpl>(
        patterns.getContext());
    ctx.getOrLoadDialect<tensor::TensorDialect>()->getCanonicalizationPatterns(
        patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createRewriteBatchMatmulToMatmulPass() {
  return std::make_unique<RewriteBatchMatmulToMatmul>();
}
