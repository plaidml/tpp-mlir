//===- TileConsumerAndFuseProducers.cpp --------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Tpp/TppUtils.h"
#include "TPP/Passes.h"
#include "TPP/TransformUtils.h"
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
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

#define DEBUG_TYPE "tile-consumer-and-fuse-producers"

namespace {

static bool isMatmulLike(Operation *op) {
  if (isa_and_nonnull<linalg::MatmulOp>(op))
    return true;
  if (linalg::GenericOp maybeMatmul = dyn_cast_or_null<linalg::GenericOp>(op))
    return linalgx::utils::isBlockedMatmul(maybeMatmul);
  return false;
}

// Return true if the current linalgOp has a 'matmul-like' op has producers.
static bool hasMatmulLikeProducer(linalg::LinalgOp linalgOp) {
  for (Value operand : linalgOp->getOperands()) {
    Operation *op = operand.getDefiningOp<linalg::LinalgOp>();
    if (op && isMatmulLike(op))
      return true;
  }
  return false;
}

static bool isConvolutionLike(Operation *op) {
  if (linalg::GenericOp maybeMatmul = dyn_cast_or_null<linalg::GenericOp>(op))
    return linalgx::utils::isBlockedConvolution(op);
  return false;
}

// Return true if the current linalgOp has a 'convolution-like' op has
// producers.
static bool hasConvolutionLikeProducer(linalg::LinalgOp linalgOp) {
  for (Value operand : linalgOp->getOperands()) {
    Operation *op = operand.getDefiningOp<linalg::LinalgOp>();
    if (op && isConvolutionLike(op))
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
    // Skip tile equals 0. A '0' tile means skip this dimension.
    if (tileSizes[tileIdx] == 0)
      continue;
    // Fail if the tile is not a full tile.
    if (*sizeRangeAtIdx % tileSizes[tileIdx] != 0)
      return false;
  }
  // Candidate op is good to go.
  return true;
}

// Returns a bit vector of size number of loops of the `interfaceOp` with
// the bits corresponding to outer parallel loops set to `true`.
// Adapted from IREE: `FormDispatchRegions.cpp`.
static llvm::SmallBitVector getOuterParallelLoops(Operation *op) {
  auto interfaceOp = dyn_cast<TilingInterface>(op);
  if (!interfaceOp)
    return llvm::SmallBitVector{};

  SmallVector<utils::IteratorType> loopIteratorTypes =
      interfaceOp.getLoopIteratorTypes();
  llvm::SmallBitVector parallelLoops(loopIteratorTypes.size());
  for (auto iteratorType : llvm::enumerate(loopIteratorTypes)) {
    if (iteratorType.value() != utils::IteratorType::parallel)
      break;
    parallelLoops.set(iteratorType.index());
  }
  return parallelLoops;
}

// Return the tile sizes as bit vector.
static llvm::SmallBitVector
getTileBitVectorConfig(ArrayRef<int64_t> tileSizes,
                       size_t rootConsumerParallelLoops) {
  assert(tileSizes.size() <= rootConsumerParallelLoops);
  llvm::SmallBitVector tileConfig(rootConsumerParallelLoops, false);
  for (size_t idx : llvm::seq<size_t>(0, tileSizes.size()))
    if (tileSizes[idx] != 0)
      tileConfig.set(idx);
  return tileConfig;
}

// Returns true if `map` is an identity map with zeros, i.e. if you
// drop the result exprs that are constant zeros, the `map` will become an
// identity.
// Taken from IREE
static bool isIdentityMapWithZeros(AffineMap map) {
  if (map.getNumSymbols() != 0)
    return false;
  unsigned dimsSeen = 0;
  for (auto result : map.getResults()) {
    bool isValidExpr = TypeSwitch<AffineExpr, bool>(result)
                           .Case<AffineDimExpr>([&dimsSeen](auto dimExpr) {
                             if (dimExpr.getPosition() != dimsSeen)
                               return false;
                             dimsSeen++;
                             return true;
                           })
                           .Case<AffineConstantExpr>([](auto constExpr) {
                             return constExpr.getValue() == 0;
                           })
                           .Default([](AffineExpr) { return false; });
    if (!isValidExpr)
      return false;
  }
  return dimsSeen == map.getNumDims();
}

// Helper fuction to print a bit vector.
static void printBitVector(std::string banner,
                           const llvm::SmallBitVector &bitVector,
                           llvm::raw_ostream &os) {
  os << banner << "  ";
  for (size_t idx : llvm::seq<size_t>(0, bitVector.size())) {
    os << bitVector.test(idx);
    if (idx != bitVector.size() - 1)
      os << ", ";
  }
  os << "\n";
}

// Return true if the candidate is as parallel as the root.
static bool
matchIteratorTypes(const llvm::SmallBitVector &rootOuterParallelLoop,
                   const llvm::SmallBitVector &candidateOuterParallelLoop) {
  if (candidateOuterParallelLoop.size() < rootOuterParallelLoop.size())
    return false;
  assert(candidateOuterParallelLoop.size() >= rootOuterParallelLoop.size());
  for (size_t idx : llvm::seq<size_t>(0, rootOuterParallelLoop.size()))
    if (rootOuterParallelLoop.test(idx) &&
        !candidateOuterParallelLoop.test(idx))
      return false;
  return true;
}

// Return true if the producer and the current consumer have compatible parallel
// dimension with the root consumer.
static bool hasCompatibleOuterParallelLoops(OpOperand &operand,
                                            Operation *producer,
                                            Operation *rootConsumer,
                                            ArrayRef<int64_t> tileSizes) {
  assert(operand.get().getDefiningOp() == producer);
  Operation *consumer = operand.getOwner();
  assert(isa<linalg::LinalgOp>(producer) && "producer must be a linalgOp");
  assert(isa<linalg::LinalgOp>(consumer) && "consumer must be a linalgOp");
  linalg::LinalgOp linalgProducer = cast<linalg::LinalgOp>(producer);
  linalg::LinalgOp linalgConsumer = cast<linalg::LinalgOp>(consumer);

  llvm::SmallBitVector producerParallelLoops =
      getOuterParallelLoops(cast<TilingInterface>(producer));
  llvm::SmallBitVector consumerParallelLoops =
      getOuterParallelLoops(cast<TilingInterface>(consumer));
  llvm::SmallBitVector rootConsumerParallelLoops =
      getOuterParallelLoops(cast<TilingInterface>(rootConsumer));
  llvm::SmallBitVector tileConfig =
      getTileBitVectorConfig(tileSizes, rootConsumerParallelLoops.size());

  LLVM_DEBUG(printBitVector("PRODUCER LOOP CONFIG", producerParallelLoops,
                            llvm::dbgs());
             printBitVector("CONSUMER LOOP CONFIG", consumerParallelLoops,
                            llvm::dbgs());
             printBitVector("ROOT CONSUMER LOOP CONFIG",
                            rootConsumerParallelLoops, llvm::dbgs());
             printBitVector("TILE CONFIG", tileConfig, llvm::dbgs()));

  producerParallelLoops &= tileConfig;
  consumerParallelLoops &= tileConfig;
  rootConsumerParallelLoops &= tileConfig;

  if (!matchIteratorTypes(rootConsumerParallelLoops, producerParallelLoops) ||
      !matchIteratorTypes(rootConsumerParallelLoops, consumerParallelLoops))
    return false;

  LLVM_DEBUG(printBitVector("PRODUCER LOOP TILE CONFIG", producerParallelLoops,
                            llvm::dbgs());
             printBitVector("CONSUMER LOOP TILE CONFIG", consumerParallelLoops,
                            llvm::dbgs());
             printBitVector("ROOT CONSUMER TILE LOOP CONFIG",
                            rootConsumerParallelLoops, llvm::dbgs()));

  auto producerIndexingMap = linalgProducer.getIndexingMapMatchingResult(
      operand.get().cast<OpResult>());
  auto consumerIndexingMap = linalgConsumer.getMatchingIndexingMap(&operand);
  if (!producerIndexingMap.isProjectedPermutation() ||
      !consumerIndexingMap.isProjectedPermutation()) {
    return false;
  }

  LLVM_DEBUG(llvm::dbgs() << "PRODUCER MAP: " << producerIndexingMap << "\n";
             llvm::dbgs() << "CONSUMER MAP: " << consumerIndexingMap << "\n";);

  producerParallelLoops.flip();
  auto producerProjectedMap =
      getProjectedMap(producerIndexingMap, producerParallelLoops);

  consumerParallelLoops.flip();
  auto consumerProjectedMap =
      getProjectedMap(consumerIndexingMap, consumerParallelLoops);

  LLVM_DEBUG(
      llvm::dbgs() << "PRODUCER PROJ MAP: " << producerProjectedMap << "\n";
      llvm::dbgs() << "CONSUMER PROJ MAP: " << consumerProjectedMap << "\n";);

  return isIdentityMapWithZeros(producerProjectedMap) &&
         isIdentityMapWithZeros(consumerProjectedMap);
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
    TilingInterface rootConsumer, ArrayRef<int64_t> tileSizes,
    const llvm::SmallDenseSet<Operation *> &alreadyFusedOps) {
  if (alreadyFusedOps.count(rootConsumer.getOperation()))
    return {};
  if (!canBeTiledWithCurrentSpec(rootConsumer, tileSizes))
    return {};

  llvm::SmallDenseSet<Operation *> worklist;
  SmallVector<Operation *> processingQueue;
  processingQueue.push_back(rootConsumer);
  worklist.insert(rootConsumer);
  LLVM_DEBUG(llvm::dbgs() << "WORKLIST INSERT CONSUMER: "
                          << rootConsumer.getOperation() << "\n");

  while (!processingQueue.empty()) {
    Operation *currentOp = processingQueue.pop_back_val();
    for (OpOperand &operand : currentOp->getOpOperands()) {
      Operation *producer = operand.get().getDefiningOp();
      if (!producer || !isa<linalg::LinalgOp>(producer) ||
          worklist.count(producer) || producer->getNumResults() != 1 ||
          alreadyFusedOps.count(producer) ||
          !hasCompatibleOuterParallelLoops(operand, producer, rootConsumer,
                                           tileSizes) ||
          !hasAllUsersInWorklist(producer, worklist))
        continue;
      LLVM_DEBUG(llvm::dbgs()
                 << "WORKLIST INSERT PRODUCER: " << producer << "\n");
      processingQueue.push_back(producer);
      worklist.insert(producer);
    }
  }
  return worklist;
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

// TODO: tileSizes should be OpFoldResult.
// Entry point for fusion.
static FailureOr<scf::SCFTileAndFuseResult>
fuseWithEltwise(RewriterBase &rewriter, TilingInterface consumer,
                ArrayRef<int64_t> tileSizes,
                llvm::SmallDenseSet<Operation *> &alreadyFusedOps) {
  // Step 0. Early exit if tileSizes are empty.
  if (tileSizes.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "EMPTY TILE SIZES\n");
    return failure();
  }

  // Step 1. If the consumer is already tiled and fused, bail out.
  if (alreadyFusedOps.count(consumer)) {
    LLVM_DEBUG(llvm::dbgs()
               << "CONSUMER: " << consumer << "\nALREADY TILED AND FUSED\n");
    return failure();
  }

  // Step 2. Check if the tile configuration fits the consumer.
  if (!canBeTiledWithCurrentSpec(consumer, tileSizes)) {
    LLVM_DEBUG(llvm::dbgs() << "CONSUMER: " << consumer
                            << "\nCANNOT BE TILED WITH CURRENT CONFIG\n");
    return failure();
  }

  // Step 3. Collect the operations that can be tiled and fused.
  llvm::SmallDenseSet<Operation *> tiledAndFusedOpCandidates =
      collectTiledAndFusedOps(consumer, tileSizes, alreadyFusedOps);
  LLVM_DEBUG(llvm::dbgs() << "#WORKLIST: " << tiledAndFusedOpCandidates.size()
                          << "\n");
  if (tiledAndFusedOpCandidates.size() == 1)
    return failure();

  // Step 4. Tile the consumer.
  scf::SCFTileAndFuseResult tileAndFuseResult;
  FailureOr<scf::SCFTilingResult> tilingResult =
      tileConsumer(rewriter, consumer, tileSizes);
  if (failed(tilingResult))
    return rewriter.notifyMatchFailure(consumer,
                                       "failed to tile base operation");
  for (auto *tiledOp : tilingResult->tiledOps) {
    tileAndFuseResult.tiledAndFusedOps.insert(tiledOp);
    alreadyFusedOps.insert(tiledOp);
    LLVM_DEBUG(llvm::dbgs() << "NEW OP: " << tiledOp << "\n");
  }
  tileAndFuseResult.loops = std::move(tilingResult->loops);
  for (const auto &result : llvm::enumerate(llvm::zip_equal(
           consumer->getResults(), tilingResult->replacements))) {
    tileAndFuseResult.replacements[std::get<0>(result.value())] =
        std::get<1>(result.value());
  }

  // If there are no loops generated (i.e., tile sizes all zeros), exit.
  if (tileAndFuseResult.loops.empty())
    return tileAndFuseResult;

  // Step 5. Tile producers and fuse into the tiled consumer.
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
    // If the op is already fused to not fuse again.
    if (alreadyFusedOps.count(candidateOp))
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
      if (alreadyFusedOps.count(consumer.getOperation()) == 0) {
        alreadyFusedOps.insert(consumer.getOperation());
        LLVM_DEBUG(llvm::dbgs()
                   << "FUSED INSERT: " << consumer.getOperation() << "\n");
      }
      Operation *untiledProducer = fusedProducer->origProducer.getDefiningOp();
      alreadyFusedOps.insert(untiledProducer);
      LLVM_DEBUG(llvm::dbgs() << "FUSED INSERT: " << untiledProducer << "\n");
      alreadyFusedOps.insert(tiledAndFusedOp);
      LLVM_DEBUG(llvm::dbgs() << "NEW OP: " << tiledAndFusedOp << "\n");
    }
  }
  return tileAndFuseResult;
}

static SmallVector<int64_t> getDefaultTileSizes() { return {32, 32}; }

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

    // Set to keep track of fused ops.
    llvm::SmallDenseSet<Operation *> fusedOps;

    func->walk([&](linalg::LinalgOp linalgOp) {
      if (linalg::isElementwise(linalgOp) &&
          (hasMatmulLikeProducer(linalgOp) ||
           hasConvolutionLikeProducer(linalgOp))) {
        if (tileSizes.empty())
          tileSizes = getDefaultTileSizes();
        LLVM_DEBUG(llvm::dbgs() << "\n\n");
        FailureOr<scf::SCFTileAndFuseResult> fuseAndTileResult =
            fuseWithEltwise(rewriter,
                            cast<TilingInterface>(linalgOp.getOperation()),
                            tileSizes, fusedOps);
        LLVM_DEBUG(llvm::dbgs() << "\n\n");
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
