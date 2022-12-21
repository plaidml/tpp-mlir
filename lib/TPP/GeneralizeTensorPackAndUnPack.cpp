//===- GeneralizeTensorPackAndUnPack.cpp -------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
//#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
//#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "TPP/Dialect/LinalgX/LinalgXOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

#if 0
struct TilePackToUnitDims : OpRewritePattern<tensor::PackOp> {
  using OpRewritePattern::OpRewritePattern;

  bool isAlreadyTiled(tensor::PackOp packOp) const {
    int64_t srcRank = packOp.getSourceRank();
    return !llvm::any_of(packOp.getDestType().getShape().take_front(srcRank),
                         [](int64_t val) { return val != 1; });
  }

  LogicalResult matchAndRewrite(tensor::PackOp packOp,
                                PatternRewriter &rewriter) const override {
    auto tilingInterfaceOp = dyn_cast<TilingInterface>(packOp.getOperation());
    if ((!tilingInterfaceOp) || (isAlreadyTiled(packOp)))
      return failure();
    SmallVector<int64_t> tiles(packOp.getSourceType().getRank(), 1);
    scf::SCFTilingOptions tilingOptions;
    tilingOptions.setTileSizes(tiles);
    FailureOr<scf::SCFTilingResult> tilingResult =
        tileUsingSCFForOp(rewriter, tilingInterfaceOp, tilingOptions);
    if (failed(tilingResult))
      return failure();
    rewriter.replaceOp(tilingInterfaceOp, tilingResult->replacements);
    return success();
  }
};

struct TileUnPackToUnitDims : OpRewritePattern<tensor::UnPackOp> {
  using OpRewritePattern::OpRewritePattern;

  bool isAlreadyTiled(tensor::UnPackOp unPackOp) const {
    int64_t destRank = unPackOp.getDestRank();
    ArrayRef<int64_t> srcShape = unPackOp.getSourceType().getShape();
    return !llvm::any_of(srcShape.take_front(destRank),
                         [](int64_t val) { return val != 1; });
  }

  LogicalResult matchAndRewrite(tensor::UnPackOp unPackOp,
                                PatternRewriter &rewriter) const override {
    auto tilingInterfaceOp = dyn_cast<TilingInterface>(unPackOp.getOperation());
    if ((!tilingInterfaceOp) || (isAlreadyTiled(unPackOp)))
      return failure();
    ShapedType sourceType = unPackOp.getSourceType();
    int64_t destRank = unPackOp.getDestType().getRank();
    SmallVector<int64_t> tiles =
        llvm::to_vector(sourceType.getShape().take_back(destRank));
    scf::SCFTilingOptions tilingOptions;
    tilingOptions.setTileSizes(tiles);
    FailureOr<scf::SCFTilingResult> tilingResult =
        tileUsingSCFForOp(rewriter, tilingInterfaceOp, tilingOptions);
    if (failed(tilingResult))
      return failure();
    rewriter.replaceOp(tilingInterfaceOp, tilingResult->replacements);
    return success();
  }
};

void populateGeneralizeTensorPackAndUnPack(RewritePatternSet &patterns) {
  patterns.insert<TilePackToUnitDims, TileUnPackToUnitDims,
                  mlir::linalg::GeneralizeOuterUnitDimsPackOpPattern,
                  mlir::linalg::GeneralizeOuterUnitDimsUnPackOpPattern>(
      patterns.getContext());
}
#endif

struct SwapWithLinalgxPack : OpRewritePattern<tensor::PackOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::PackOp packOp,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<linalgx::PackOp>(
        packOp, packOp.getSource(), packOp.getDest(), packOp.getInnerDimsPos(),
        packOp.getMixedTiles(), packOp.getPaddingValue(),
        packOp.getOuterDimsPerm());
    return success();
  }
};

struct SwapWithLinalgxUnPack : OpRewritePattern<tensor::UnPackOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::UnPackOp unPackOp,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<linalgx::UnPackOp>(
        unPackOp, unPackOp.getSource(), unPackOp.getDest(),
        unPackOp.getInnerDimsPos(), unPackOp.getOuterDimsPerm(),
        unPackOp.getMixedTiles());
    return success();
  }
};

void populateGeneralizeTensorPackAndUnPack(RewritePatternSet &patterns) {
  patterns.insert<SwapWithLinalgxPack, SwapWithLinalgxUnPack>(
      patterns.getContext());
}

struct GeneralizeTensorPackAndUnPack
    : public GeneralizeTensorPackAndUnPackBase<GeneralizeTensorPackAndUnPack> {
  GeneralizeTensorPackAndUnPack() = default;

  void runOnOperation() override {
    RewritePatternSet patterns(getOperation().getContext());
    populateGeneralizeTensorPackAndUnPack(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // end namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createGeneralizeTensorPackAndUnPackPass() {
  return std::make_unique<GeneralizeTensorPackAndUnPack>();
}
