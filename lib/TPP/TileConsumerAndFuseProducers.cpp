//===- TileConsumerAndFuseProducers.cpp --------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Tpp/TppUtils.h"
#include "TPP/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/IR/TransformUtils.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

// TODO: improve checks here.
static bool isBlockedMatmul(linalg::GenericOp genericOp) {
  return tpp::utils::hasMatmulBody(genericOp);
}

static bool isMatmulLike(Operation *op) {
  if (isa_and_nonnull<linalg::MatmulOp>(op))
    return true;
  if (linalg::GenericOp maybeMatmul = dyn_cast_or_null<linalg::GenericOp>(op))
    return isBlockedMatmul(maybeMatmul);
  return false;
}

static bool hasMatmulLikeProducer(linalg::LinalgOp linalgOp) {
  for (Value operand : linalgOp->getOperands()) {
    Operation *op = operand.getDefiningOp<linalg::LinalgOp>();
    if (op && isMatmulLike(op))
      return true;
  }
  return false;
}

static FailureOr<scf::SCFTilingResult>
tileConsumer(RewriterBase &rewriter, TilingInterface consumer,
             ArrayRef<int64_t> tileSizes) {
  // If tileSizes is provided use them otherwise default to 1, 1, meaning fuse
  // entire the two outermost loops.
  SmallVector<int64_t> actualTileSizes = tileSizes.empty()
                                             ? SmallVector<int64_t>{1, 1}
                                             : llvm::to_vector(tileSizes);
  auto options = scf::SCFTilingOptions().setTileSizes(actualTileSizes);
  FailureOr<scf::SCFTilingResult> tilingResult =
      scf::tileUsingSCFForOp(rewriter, consumer, options);
  return tilingResult;
}

// Tile and fuse a matmul along the parallel dimensions.
static FailureOr<scf::SCFTileAndFuseResult>
fuseMatmulLikeAndEltwise(RewriterBase &rewriter, TilingInterface consumer,
                         ArrayRef<int64_t> tileSizes) {

  // Step 1. tile the consumer.
  scf::SCFTileAndFuseResult tileAndFuseResult;
  FailureOr<scf::SCFTilingResult> tilingResult =
      tileConsumer(rewriter, consumer, tileSizes);
  if (failed(tilingResult))
    return rewriter.notifyMatchFailure(consumer,
                                       "failed to tile base operation");
  for (auto *tiledOp : tilingResult->tiledOps)
    tileAndFuseResult.tiledAndFusedOps.insert(tiledOp);
  tileAndFuseResult.loops = std::move(tilingResult->loops);
  for (const auto &result : llvm::enumerate(
           llvm::zip(consumer->getResults(), tilingResult->replacements))) {
    tileAndFuseResult.replacements[std::get<0>(result.value())] =
        std::get<1>(result.value());
  }

  // Step 2. tile producers and fuse into the tiled consumer.
  auto addCandidateSlices = [](Operation *fusedOp,
                               std::deque<tensor::ExtractSliceOp> &candidates) {
    for (Value operand : fusedOp->getOperands())
      if (auto sliceOp = operand.getDefiningOp<tensor::ExtractSliceOp>())
        candidates.push_back(sliceOp);
  };

  std::deque<tensor::ExtractSliceOp> candidates;
  addCandidateSlices(tilingResult->tiledOps.back(), candidates);
  OpBuilder::InsertionGuard g(rewriter);
  while (!candidates.empty()) {
    tensor::ExtractSliceOp candidateSliceOp = candidates.front();
    candidates.pop_front();

    std::optional<scf::SCFFuseProducerOfSliceResult> fusedProducer =
        tileAndFuseProducerOfSlice(rewriter, candidateSliceOp,
                                   tileAndFuseResult.loops);
    if (!fusedProducer)
      continue;

    if (Operation *tiledAndFusedOp =
            fusedProducer->tiledAndFusedProducer.getDefiningOp()) {
      tileAndFuseResult.tiledAndFusedOps.insert(tiledAndFusedOp);
      addCandidateSlices(tiledAndFusedOp, candidates);
    }
  }

  return tileAndFuseResult;
}

struct TileConsumerAndFuseProducers
    : TileConsumerAndFuseProducersBase<TileConsumerAndFuseProducers> {
  TileConsumerAndFuseProducers() = default;
  TileConsumerAndFuseProducers(ArrayRef<int64_t> tileSizes) {
    this->tileSizes = tileSizes;
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
    linalg::registerTilingInterfaceExternalModels(registry);
  }
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    transform::TrivialPatternRewriter rewriter(&getContext());
    func->walk([&](linalg::LinalgOp linalgOp) {
      if (linalg::isElementwise(linalgOp) && hasMatmulLikeProducer(linalgOp)) {
        FailureOr<scf::SCFTileAndFuseResult> fuseAndTileResult =
            fuseMatmulLikeAndEltwise(
                rewriter, cast<TilingInterface>(linalgOp.getOperation()),
                this->tileSizes);
        if (failed(fuseAndTileResult))
          return signalPassFailure();
        rewriter.replaceOp(
            linalgOp,
            (*fuseAndTileResult).replacements[linalgOp->getResults()[0]]);
      }
    });
    RewritePatternSet patterns(&getContext());
    // fold unit-extent dims for linalg on tensors.
    linalg::populateFoldUnitExtentDimsViaSlicesPatterns(patterns);
    tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // end namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createTileConsumerAndFuseProducersPass() {
  return std::make_unique<TileConsumerAndFuseProducers>();
}
