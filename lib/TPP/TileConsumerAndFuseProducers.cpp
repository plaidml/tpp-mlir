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
#include "llvm/Support/Debug.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

#define DEBUG_TYPE "tile-consumer-and-fuse-producers"

namespace {

// TODO: Check indexing maps and iterator types. They should
// match the one of a packed matmul.
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

// Return the range as int64_t for `range` if we can statically compute it,
// otherwise return std::nullopt.
static std::optional<int64_t> getSizeRangeAtIdx(const Range &range) {
  std::optional<int64_t> stride = getConstantIntValue(range.stride);
  if (!stride || *stride != 1)
    return std::nullopt;
  std::optional<int64_t> offset = getConstantIntValue(range.offset);
  if (!offset)
    return std::nullopt;
  std::optional<int64_t> size = getConstantIntValue(range.size);
  if (!size)
    return std::nullopt;
  return (*size - *offset);
}

// Return true if `op` can be tiled using `tileSizes`. Require to statically
// know the range and the tile factor. The tile must be full.
static bool canBeTiledWithCurrentSpec(Operation *op,
                                      ArrayRef<int64_t> tileSizes) {
  assert(isa<TilingInterface>(op) &&
         "expect an op implementing the tiling interface");
  assert(!tileSizes.empty() && "expect tile sizes to be non-empty");
  OpBuilder builder(op);
  SmallVector<Range> iterationDomain =
      cast<TilingInterface>(op).getIterationDomain(builder);
  SmallVector<utils::IteratorType> loopIteratorTypes =
      cast<TilingInterface>(op).getLoopIteratorTypes();
  if (tileSizes.size() > iterationDomain.size())
    return false;
  for (size_t tileIdx = 0, idxEnd = tileSizes.size(); tileIdx < idxEnd;
       tileIdx++) {
    // Allow tiling only on parallel loop iterators.
    if (!linalg::isParallelIterator(loopIteratorTypes[tileIdx]))
      return false;
    std::optional<int64_t> sizeRangeAtIdx =
        getSizeRangeAtIdx(iterationDomain[tileIdx]);
    // Non constant range. Bail out and do not add the op as candidate.
    if (!sizeRangeAtIdx)
      return false;
    // Corner case tile equals 0.
    if (tileSizes[tileIdx] == 0)
      return false;
    // Fail if the tile size equals the range or it is not a full tile.
    if (tileSizes[tileIdx] == *sizeRangeAtIdx ||
        *sizeRangeAtIdx % tileSizes[tileIdx] != 0)
      return false;
  }
  // Candidate op is good to go.
  return true;
}

// Tile the consumer and return the tiled result.
static FailureOr<scf::SCFTilingResult>
tileConsumer(RewriterBase &rewriter, TilingInterface consumer,
             ArrayRef<int64_t> tileSizes) {
  assert(!tileSizes.empty() && "expect tile sizes to be non-empty");
  assert(canBeTiledWithCurrentSpec(consumer, tileSizes) &&
         "expect valid tile sizes");
  auto options = scf::SCFTilingOptions().setTileSizes(tileSizes);
  FailureOr<scf::SCFTilingResult> tilingResult =
      scf::tileUsingSCFForOp(rewriter, consumer, options);
  return tilingResult;
}

// Return true if the op has all users in the worklist. This is an important
// condition as we want to fuse together 'isolated islands' of op. Basically
// each op in the worklist must have users in the worklist. If this is not the
// case we can introduce recomputation.
bool hasAllUsersInWorklist(Operation *op,
                           const llvm::SmallDenseSet<Operation *> &worklist) {
  assert(op->getNumResults() == 1 && "expect single result op");
  Value result = op->getResult(0);
  for (Operation *user : result.getUsers())
    if (worklist.count(user) == 0)
      return false;
  return true;
}

// Return a list of op that can be fused together based on what has already been
// fused and the current tile specification.
static llvm::SmallDenseSet<Operation *> collectTiledAndFusedOps(
    TilingInterface consumer, ArrayRef<int64_t> tileSizes,
    const llvm::SmallDenseSet<Operation *> &alreadyFusedOps) {
  if (alreadyFusedOps.count(consumer.getOperation()))
    return {};

  llvm::SmallDenseSet<Operation *> fusableProducers;
  SmallVector<Operation *> worklist;
  worklist.push_back(consumer);
  LLVM_DEBUG(llvm::dbgs() << "WORKLIST INSERT CONSUMER: "
                          << consumer.getOperation() << "\n");
  fusableProducers.insert(consumer);

  while (!worklist.empty()) {
    Operation *currentOp = worklist.pop_back_val();
    for (OpOperand &operand : currentOp->getOpOperands()) {
      Operation *producer = operand.get().getDefiningOp();
      if (!producer || !isa<linalg::LinalgOp>(producer) ||
          fusableProducers.count(producer) ||
          !canBeTiledWithCurrentSpec(producer, tileSizes) ||
          alreadyFusedOps.count(producer) || producer->getNumResults() != 1 ||
          !hasAllUsersInWorklist(producer, fusableProducers))
        continue;
      LLVM_DEBUG(llvm::dbgs()
                 << "WORKLIST INSERT PRODUCER: " << producer << "\n");
      worklist.push_back(producer);
      fusableProducers.insert(producer);
    }
  }
  return fusableProducers;
}

// Walk source and loops and return the defining op of 'source' if it exists.
static Operation *
getUntiledProducerFromSliceSource(OpOperand *source,
                                  ArrayRef<scf::ForOp> loops) {
  auto loopIt = loops.rbegin();
  while (auto iterArg = source->get().dyn_cast<BlockArgument>()) {
    scf::ForOp loop = *loopIt;
    if (iterArg.getOwner()->getParentOp() != loop)
      break;
    source = &loop.getOpOperandForRegionIterArg(iterArg);
    loopIt++;
  }
  return source->get().getDefiningOp();
}

// Tile and fuse a matmul along the parallel dimensions.
static FailureOr<scf::SCFTileAndFuseResult>
fuseMatmulLikeAndEltwise(RewriterBase &rewriter, TilingInterface consumer,
                         ArrayRef<int64_t> tileSizes,
                         llvm::SmallDenseSet<Operation *> &alreadyFusedOps) {
  // Step -1. Check if the tile configuration fits the consumer.
  if (!canBeTiledWithCurrentSpec(consumer, tileSizes))
    return failure();

  // Step 0. Collect the operations that can be tiled and fused.
  llvm::SmallDenseSet<Operation *> tiledAndFusedOpCandidates =
      collectTiledAndFusedOps(consumer, tileSizes, alreadyFusedOps);
  LLVM_DEBUG(llvm::dbgs() << "#WORKLIST: " << tiledAndFusedOpCandidates.size()
                          << "\n");
  if (tiledAndFusedOpCandidates.size() == 1)
    return failure();

  // Step 1. Tile the consumer.
  scf::SCFTileAndFuseResult tileAndFuseResult;
  FailureOr<scf::SCFTilingResult> tilingResult =
      tileConsumer(rewriter, consumer, tileSizes);
  if (failed(tilingResult))
    return rewriter.notifyMatchFailure(consumer,
                                       "failed to tile base operation");
  for (auto *tiledOp : tilingResult->tiledOps)
    tileAndFuseResult.tiledAndFusedOps.insert(tiledOp);
  tileAndFuseResult.loops = std::move(tilingResult->loops);
  for (const auto &result : llvm::enumerate(llvm::zip_equal(
           consumer->getResults(), tilingResult->replacements))) {
    tileAndFuseResult.replacements[std::get<0>(result.value())] =
        std::get<1>(result.value());
  }

  // Step 2. Tile producers and fuse into the tiled consumer.
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

    // Find the candidate operation potentially walking bbArgs in scf.for.
    // If we find a candidate we check if it is in our worklist and fuse it
    // only if so.
    Operation *candidateOp = getUntiledProducerFromSliceSource(
        &candidateSliceOp->getOpOperand(0), tileAndFuseResult.loops);
    if (!candidateOp || tiledAndFusedOpCandidates.count(candidateOp) == 0)
      continue;

    std::optional<scf::SCFFuseProducerOfSliceResult> fusedProducer =
        tileAndFuseProducerOfSlice(rewriter, candidateSliceOp,
                                   tileAndFuseResult.loops);
    if (!fusedProducer)
      continue;

    if (Operation *tiledAndFusedOp =
            fusedProducer->tiledAndFusedProducer.getDefiningOp()) {
      tileAndFuseResult.tiledAndFusedOps.insert(tiledAndFusedOp);
      addCandidateSlices(tiledAndFusedOp, candidates);
      alreadyFusedOps.insert(consumer.getOperation());
      Operation *untiledProducer = fusedProducer->origProducer.getDefiningOp();
      alreadyFusedOps.insert(untiledProducer);
      LLVM_DEBUG(llvm::dbgs() << "FUSED INSERT: " << untiledProducer << "\n");
      LLVM_DEBUG(llvm::dbgs()
                 << "FUSED INSERT: " << consumer.getOperation() << "\n");
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
    llvm::SmallDenseSet<Operation *> fusedOps;
    func->walk([&](linalg::LinalgOp linalgOp) {
      if (linalg::isElementwise(linalgOp) && hasMatmulLikeProducer(linalgOp)) {
        SmallVector<int64_t> actualTileSizes = llvm::to_vector(this->tileSizes);
        // Default tiling scheme, always legal to tile by 1.
        if (actualTileSizes.empty())
          actualTileSizes = {1, 1};
        FailureOr<scf::SCFTileAndFuseResult> fuseAndTileResult =
            fuseMatmulLikeAndEltwise(
                rewriter, cast<TilingInterface>(linalgOp.getOperation()),
                actualTileSizes, fusedOps);
        if (succeeded(fuseAndTileResult)) {
          rewriter.replaceOp(
              linalgOp,
              (*fuseAndTileResult).replacements[linalgOp->getResults()[0]]);
        }
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
