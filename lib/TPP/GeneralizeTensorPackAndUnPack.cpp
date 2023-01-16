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
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/IR/TransformUtils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

struct GeneralizeTensorPackAndUnPack
    : public GeneralizeTensorPackAndUnPackBase<GeneralizeTensorPackAndUnPack> {
  GeneralizeTensorPackAndUnPack() = default;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    transform::TrivialPatternRewriter rewriter(&getContext());
    func->walk([&](tensor::UnPackOp unPackOp) {
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
    func->walk([&](tensor::PackOp packOp) {
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
    return;
  }
};

} // end namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createGeneralizeTensorPackAndUnPackPass() {
  return std::make_unique<GeneralizeTensorPackAndUnPack>();
}
