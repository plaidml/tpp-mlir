//===- GeneralizeTensorPackAndUnPack.cpp -------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "decompose-pack-unpack-ops"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

// A warpper pattern that calls linalg::lowerPack on tensor::PackOp. It lowers
// a tensor.pack op to tensor.pad + tensor.expand_shape + linalg.transpose ops.
struct LowerPackPattern : public OpRewritePattern<tensor::PackOp> {
  using OpRewritePattern<tensor::PackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PackOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<linalg::LowerPackResult> res = linalg::lowerPack(rewriter, op);
    if (failed(res)) {
      return rewriter.notifyMatchFailure(
          op, "cannot lower to pad + expand + transpose");
    }
    return success();
  }
};

// A warpper pattern that calls linalg::lowerUnPack on tensor::UnPackOp. It
// lowers a tensor.unpack op to tensor.empty + linalg.transpose +
// tensor.collapse_shape + tensor.extract_slice ops.
struct LowerUnPackPattern : public OpRewritePattern<tensor::UnPackOp> {
  using OpRewritePattern<tensor::UnPackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::UnPackOp op,
                                PatternRewriter &rewriter) const override {
    if (failed(linalg::lowerUnPack(rewriter, op))) {
      return rewriter.notifyMatchFailure(
          op, "cannot lower to empty + transpose + reshape + extract_slice");
    }
    return success();
  }
};

struct GeneralizeTensorPackAndUnPack
    : public GeneralizeTensorPackAndUnPackBase<GeneralizeTensorPackAndUnPack> {
  GeneralizeTensorPackAndUnPack() = default;
  void runOnOperation() override {

    MLIRContext *ctx = &getContext();
    auto funcOp = getOperation();

    // Upstream generalization patterns. Decomposition of tensor.unpack
    // does not support yet outer dim perm.
    {
      RewritePatternSet patterns(ctx);
      patterns.add<LowerPackPattern, LowerUnPackPattern>(ctx);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        funcOp.emitError(
            "failed to apply generalization patterns on pack/unpack ops for "
            "general cases.");
        return signalPassFailure();
      }
    }

    // Fall back on tile by one + generalization patterns.
    {
      IRRewriter rewriter(&getContext());
      funcOp->walk([&](tensor::UnPackOp unPackOp) {
        scf::SCFTilingOptions unpackTilingOptions;
        SmallVector<int64_t> tiles(unPackOp.getDestType().getRank(), 1);
        unpackTilingOptions.setTileSizes(tiles);
        FailureOr<scf::SCFTilingResult> tilingResult = scf::tileUsingSCFForOp(
            rewriter, cast<TilingInterface>(unPackOp.getOperation()),
            unpackTilingOptions);
        if (failed(tilingResult))
          return signalPassFailure();
        rewriter.replaceOp(unPackOp, tilingResult->replacements);
      });
      funcOp->walk([&](tensor::PackOp packOp) {
        SmallVector<int64_t> tiles(packOp.getSourceType().getRank(), 1);
        scf::SCFTilingOptions packTilingOptions;
        packTilingOptions.setTileSizes(tiles);
        FailureOr<scf::SCFTilingResult> tilingResult = scf::tileUsingSCFForOp(
            rewriter, cast<TilingInterface>(packOp.getOperation()),
            packTilingOptions);
        if (failed(tilingResult))
          return signalPassFailure();
        rewriter.replaceOp(packOp, tilingResult->replacements);
      });
      RewritePatternSet patterns(&getContext());
      patterns.add<linalg::GeneralizeOuterUnitDimsUnPackOpPattern,
                   linalg::GeneralizeOuterUnitDimsPackOpPattern>(&getContext());
      tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    // Canonicalize tiled ops.
    {
      RewritePatternSet patterns(ctx);
      linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
      ctx->getOrLoadDialect<tensor::TensorDialect>()
          ->getCanonicalizationPatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After canonicalizing tiled ops ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }
};

} // end namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createGeneralizeTensorPackAndUnPackPass() {
  return std::make_unique<GeneralizeTensorPackAndUnPack>();
}
