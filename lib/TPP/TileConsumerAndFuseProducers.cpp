//===- TileConsumerAndFuseProducers.cpp --------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"
#include "TPP/TransformUtils.h"
#include "TPP/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <queue>

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

#define DEBUG_TYPE "tile-consumer-and-fuse-producers"

namespace {

// Replace the iter operand of the outermost loop with the region iter argument
// of the innermost loop in the region of the innermost loop. This fix-up
// destination passing style in tile-consumer-and-fuse-producers:
// https://github.com/llvm/llvm-project/issues/61386
struct ReplaceIterArgs : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  void replaceIterArgs(PatternRewriter &rewriter, scf::ForOp outerFor,
                       scf::ForOp innerFor) const {
    assert(outerFor.getNumIterOperands() == innerFor.getNumIterOperands() &&
           "expect same number of iter args");
    Block *block = &(*innerFor.getRegion().begin());
    for (auto it : llvm::zip_equal(outerFor.getIterOperands(),
                                   innerFor.getRegionIterArgs())) {
      Value source = std::get<0>(it);
      Value target = std::get<1>(it);
      rewriter.replaceUsesWithIf(source, target, [&](OpOperand &use) {
        return use.getOwner()->getBlock() == block;
      });
    }
  }

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    auto metadata =
        forOp->getAttrOfType<StringAttr>(linalgx::utils::kLoopParallel);
    if (!metadata || metadata.getValue() != linalgx::utils::kLoopRoot)
      return failure();
    if (forOp.getNumRegionIterArgs() != 1)
      return failure();
    SmallVector<scf::ForOp> nestedLoops;
    getPerfectlyNestedLoops(nestedLoops, forOp);
    if (nestedLoops.size() == 0)
      return failure();
    replaceIterArgs(rewriter, forOp, nestedLoops[nestedLoops.size() - 1]);
    return success();
  }
};

static bool isMatmulLike(Operation *op) {
  return linalgx::utils::isBlockedMatmul(op) || isa<linalg::MatmulOp>(op);
}

static bool isConvolutionLike(Operation *op) {
  if (linalg::GenericOp maybeMatmul = dyn_cast_or_null<linalg::GenericOp>(op))
    return linalgx::utils::isBlockedConvolution(op);
  return false;
}

// Return true if `op` can be tiled using `tileSizes`. Require to statically
// know the range and the tile factor. The tile must be full.
static bool canBeTiledWithCurrentSpec(Operation *op,
                                      ArrayRef<OpFoldResult> tileSizes) {
  assert(isa<TilingInterface>(op) &&
         "expect an op implementing the tiling interface");
  assert(!tileSizes.empty() && "expect tile sizes to be non-empty");
  SmallVector<utils::IteratorType> loopIteratorTypes =
      cast<TilingInterface>(op).getLoopIteratorTypes();
  if (tileSizes.size() > loopIteratorTypes.size())
    return false;

  if (!linalgx::utils::validateFullTilesOnDims(cast<TilingInterface>(op),
                                               tileSizes))
    return false;

  // Require tiling on parallel dimensions only.
  for (auto tileIdx : llvm::seq<size_t>(0, tileSizes.size()))
    if (!linalg::isParallelIterator(loopIteratorTypes[tileIdx]))
      return false;

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
getTileBitVectorConfig(ArrayRef<OpFoldResult> tileSizes,
                       size_t rootConsumerParallelLoops) {
  assert(tileSizes.size() <= rootConsumerParallelLoops);
  llvm::SmallBitVector tileConfig(rootConsumerParallelLoops, false);
  for (size_t idx : llvm::seq<size_t>(0, tileSizes.size()))
    if (!isConstantIntValue(tileSizes[idx], 0))
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
                                            ArrayRef<OpFoldResult> tileSizes) {
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
             ArrayRef<OpFoldResult> tileSizes) {
  assert(!tileSizes.empty() && "expect tile sizes to be non-empty");
  assert(canBeTiledWithCurrentSpec(consumer, tileSizes) &&
         "expect valid tile sizes");
  auto tileSizeComputationFunction = [tileSizes](OpBuilder &builder,
                                                 Operation *op) {
    OpBuilder::InsertionGuard guard(builder);
    return getValueOrCreateConstantIndexOp(builder, op->getLoc(), tileSizes);
  };
  auto options = scf::SCFTilingOptions().setTileSizeComputationFunction(
      tileSizeComputationFunction);

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

void incDepthAndSwap(std::queue<Operation *> &frontier,
                     std::queue<Operation *> &nextFrontier, int64_t &depth) {
  if (frontier.empty() && !nextFrontier.empty()) {
    std::swap(frontier, nextFrontier);
    depth++;
  }
}

// Return a list of producers op that can be fused together based on what has
// already been fused and the current tile specification.
static llvm::SmallDenseSet<Operation *> collectFusableProducers(
    TilingInterface rootConsumer, ArrayRef<OpFoldResult> tileSizes,
    const llvm::SmallDenseSet<Operation *> &alreadyFusedOps, int64_t maxDepth) {
  if (alreadyFusedOps.count(rootConsumer.getOperation()))
    return {};
  if (!canBeTiledWithCurrentSpec(rootConsumer, tileSizes))
    return {};

  llvm::SmallDenseSet<Operation *> worklist;
  std::queue<Operation *> processingQueue;
  std::queue<Operation *> nextProcessingQueue;
  processingQueue.push(rootConsumer);
  worklist.insert(rootConsumer);
  LLVM_DEBUG(llvm::dbgs() << "WORKLIST INSERT CONSUMER: "
                          << rootConsumer.getOperation() << "\n");
  LLVM_DEBUG(
      llvm::dbgs()
          << "---------------------------------------------------------\n";
      llvm::dbgs() << *rootConsumer.getOperation();
      llvm::dbgs()
      << "\n---------------------------------------------------------\n";);
  int64_t depth = 0;
  while (!processingQueue.empty()) {
    if (depth >= maxDepth)
      break;
    Operation *currentOp = processingQueue.front();
    processingQueue.pop();
    for (OpOperand &operand : currentOp->getOpOperands()) {
      Operation *producer = operand.get().getDefiningOp();
      if (producer && isa<linalg::LinalgOp>(producer) &&
          !worklist.count(producer) && producer->getNumResults() == 1 &&
          !alreadyFusedOps.count(producer) &&
          hasCompatibleOuterParallelLoops(operand, producer, rootConsumer,
                                          tileSizes) &&
          hasAllUsersInWorklist(producer, worklist)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "WORKLIST INSERT PRODUCER: " << producer << "\n");
        LLVM_DEBUG(llvm::dbgs() << "=============================\n";
                   llvm::dbgs() << *producer << "\n";
                   llvm::dbgs() << "=============================\n";);
        nextProcessingQueue.push(producer);
        incDepthAndSwap(processingQueue, nextProcessingQueue, depth);
        worklist.insert(producer);
      }
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

// Entry point for fusion.
static FailureOr<scf::SCFTileAndFuseResult>
fuseWithEltwise(RewriterBase &rewriter, TilingInterface consumer,
                ArrayRef<OpFoldResult> tileSizes,
                llvm::SmallDenseSet<Operation *> &alreadyFusedOps,
                int64_t maxDepth) {
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
  llvm::SmallDenseSet<Operation *> worklist =
      collectFusableProducers(consumer, tileSizes, alreadyFusedOps, maxDepth);
  LLVM_DEBUG(llvm::dbgs() << "#WORKLIST: " << worklist.size() << "\n");
  if (worklist.size() < 2)
    return failure();

  // Step 4. Tile the consumer.
  scf::SCFTileAndFuseResult tileAndFuseResult;
  FailureOr<scf::SCFTilingResult> tilingResult =
      tileConsumer(rewriter, consumer, tileSizes);
  if (failed(tilingResult)) {
    return rewriter.notifyMatchFailure(consumer,
                                       "failed to tile base operation");
  }
  for (auto *tiledOp : tilingResult->tiledOps) {
    tileAndFuseResult.tiledAndFusedOps.insert(tiledOp);
    alreadyFusedOps.insert(tiledOp);
    LLVM_DEBUG(llvm::dbgs() << "NEW OP: " << tiledOp << "\n");
  }
  if (!tilingResult->loops.empty()) {
    tilingResult->loops[0]->setAttr(
        linalgx::utils::kLoopParallel,
        rewriter.getStringAttr(linalgx::utils::kLoopRoot));
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
                               std::queue<tensor::ExtractSliceOp> &candidates) {
    for (Value operand : fusedOp->getOperands())
      if (auto sliceOp = operand.getDefiningOp<tensor::ExtractSliceOp>())
        candidates.push(sliceOp);
  };

  std::queue<tensor::ExtractSliceOp> frontier;
  addCandidateSlices(tilingResult->tiledOps.back(), frontier);
  OpBuilder::InsertionGuard g(rewriter);
  while (!frontier.empty()) {
    tensor::ExtractSliceOp candidateSliceOp = frontier.front();
    frontier.pop();

    // Find the candidate operation potentially walking bbArgs in scf.for.
    // If we find a candidate we check if it is in our worklist and fuse it
    // only if so. We do not consider the candidate if it is already fused.
    Operation *candidateOp = getUntiledProducerFromSliceSource(
        &candidateSliceOp->getOpOperand(0), tileAndFuseResult.loops);
    if (!candidateOp || worklist.count(candidateOp) == 0 ||
        alreadyFusedOps.count(candidateOp))
      continue;

    std::optional<scf::SCFFuseProducerOfSliceResult> fusedProducer =
        tileAndFuseProducerOfSlice(rewriter, candidateSliceOp,
                                   tileAndFuseResult.loops);
    if (!fusedProducer)
      continue;

    if (Operation *tiledAndFusedOp =
            fusedProducer->tiledAndFusedProducer.getDefiningOp()) {
      tileAndFuseResult.tiledAndFusedOps.insert(tiledAndFusedOp);
      addCandidateSlices(tiledAndFusedOp, frontier);
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
      // This is for debug.
      if (auto metadata = untiledProducer->getAttr("metadata"))
        tiledAndFusedOp->setAttr("metadata", metadata);
    }
  }
  return tileAndFuseResult;
}

static SmallVector<int64_t> getDefaultTileSizes(linalg::LinalgOp linalgOp) {
  // Blocked convolutions are tiled and fused along the three outermost parallel
  // loops to expose a BRGEMM.
  if (linalgx::utils::isBlockedConvolution(linalgOp))
    return {1, 1, 1};
  // Matmuls are tiled and fused along i and j with 32.
  if (isa<linalg::MatmulOp>(linalgOp))
    return {32, 32};
  // Blocked matmuls are tiled and fused along the two outermost parallel loops
  // to expose a BRGEMM.
  if (linalgx::utils::isBlockedMatmul(linalgOp))
    return {1, 1};
  return {32, 32};
}

static Operation *
getLastFusableConsumer(linalg::LinalgOp linalgOp,
                       llvm::SmallDenseSet<Operation *> &visitedConsumers) {
  assert(linalgOp->getNumResults() == 1 && "Expect single result operation");
  Value linalgOpRes = linalgOp->getResult(0);
  // If we allow use, we may end up doing recomputation. Unclear if it is
  // profitablem thus disallow for now.
  if (!linalgOpRes.hasOneUse())
    return linalgOp;

  // Start checking consumers.
  Operation *nextConsumer = *(linalgOpRes.getUsers().begin());
  Operation *currentConsumer = linalgOp;
  visitedConsumers.insert(currentConsumer);

  auto isValidEltWiseConsumer = [&](Operation *op) {
    // Make sure we visit each consumer only once.
    if (visitedConsumers.count(op))
      return false;

    if (!isa<linalg::LinalgOp>(op) ||
        !linalg::isElementwise(cast<linalg::LinalgOp>(op))) {
      return false;
    }
    // Require same iteration space.
    if (cast<linalg::LinalgOp>(op).getNumParallelLoops() !=
        cast<linalg::LinalgOp>(currentConsumer).getNumParallelLoops())
      return false;

    return op->getNumResults() == 1;
  };

  while (isValidEltWiseConsumer(nextConsumer)) {
    Value resNextConsumer = nextConsumer->getResult(0);
    currentConsumer = nextConsumer;
    visitedConsumers.insert(currentConsumer);
    // Require each eltwise to have a single user.
    if (std::distance(resNextConsumer.getUsers().begin(),
                      resNextConsumer.getUsers().end()) != 1) {
      break;
    }
    nextConsumer = *(resNextConsumer.getUsers().begin());
  }
  LLVM_DEBUG(llvm::dbgs() << "LAST FUSABLE CONSUMER: " << currentConsumer
                          << "\n");
  return currentConsumer;
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
    auto &ctx = getContext();

    {
      // Attempt to recover named ops.
      RewritePatternSet patterns(&ctx);
      linalg::populateLinalgDeGeneralizationPatterns(patterns);
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    }

    func::FuncOp func = getOperation();
    IRRewriter rewriter(&getContext());

    // Set to keep track of fused ops.
    llvm::SmallDenseSet<Operation *> fusedOps;

    SmallVector<linalg::LinalgOp> linalgOperations;
    func->walk([&](linalg::LinalgOp linalgOp) {
      if ((isConvolutionLike(linalgOp) || isMatmulLike(linalgOp)) &&
          linalgOp.hasTensorSemantics())
        linalgOperations.push_back(linalgOp);
    });

    std::reverse(linalgOperations.begin(), linalgOperations.end());
    llvm::SmallDenseSet<Operation *> visitedConsumers;
    llvm::SmallDenseSet<Operation *> fusionRoots;
    DenseMap<Operation *, SmallVector<int64_t>> defaultTiles;
    for (linalg::LinalgOp linalgOp : linalgOperations) {
      Operation *rootOp = linalgOp.getOperation();
      if (this->startFromLastFusableConsumer)
        rootOp = getLastFusableConsumer(linalgOp, visitedConsumers);
      fusionRoots.insert(rootOp);
      defaultTiles[rootOp] = (this->tileSizes.empty())
                                 ? getDefaultTileSizes(linalgOp)
                                 : llvm::to_vector(this->tileSizes);
    }
    LLVM_DEBUG(llvm::dbgs() << "#fusionRoots: " << fusionRoots.size() << "\n");

    SmallVector<Operation *> allLinalgOps;
    func->walk(
        [&](linalg::LinalgOp linalgOp) { allLinalgOps.push_back(linalgOp); });

    // We want to walk operations in reverse order as fusion will likely find
    // larger cluster of operations to fuse if we go bottom-up.
    std::reverse(allLinalgOps.begin(), allLinalgOps.end());

    for (Operation *linalgOp : allLinalgOps) {
      if (fusionRoots.count(linalgOp)) {
        LLVM_DEBUG(llvm::dbgs() << "\n\n");
        FailureOr<scf::SCFTileAndFuseResult> fuseAndTileResult =
            fuseWithEltwise(rewriter, cast<TilingInterface>(linalgOp),
                            getAsOpFoldResult(rewriter.getI64ArrayAttr(
                                defaultTiles[linalgOp])),
                            fusedOps, this->maxDepth);
        LLVM_DEBUG(llvm::dbgs() << "\n\n");
        if (succeeded(fuseAndTileResult)) {
          rewriter.replaceOp(
              linalgOp,
              (*fuseAndTileResult).replacements[linalgOp->getResults()[0]]);
        }
      }
    }

    {
      // Patterns for scf.for.
      RewritePatternSet patterns(&ctx);
      patterns.add<ReplaceIterArgs>(&ctx);
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    }

    {
      // Patterns for scf.forall.
      RewritePatternSet patterns(&ctx);
      if (this->useForAll)
        linalgx::utils::populateScfForToForAllRewritePattern(patterns);
      // Fold unit-extent dims for linalg on tensors. Since
      // `populateFoldUnitExtentDimsViaSlicesPatterns` works only with
      // linalg.generic we need to generalize first using
      // `populateLinalgNamedOpsGeneralizationPatterns`.
      linalg::populateFoldUnitExtentDimsViaSlicesPatterns(patterns);
      tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
      linalg::populateLinalgNamedOpsGeneralizationPatterns(patterns);
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    }

    {
      // Attempt to recover named ops.
      RewritePatternSet patterns(&ctx);
      linalg::populateLinalgDeGeneralizationPatterns(patterns);
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    }
  }
};

static bool areFusableOps(Operation *producerOp, Operation *consumerOp) {
  if ((!isa<linalg::LinalgOp>(producerOp)) ||
      (!isa<linalg::LinalgOp>(consumerOp)))
    return false;
  if (!linalg::isElementwise(cast<linalg::LinalgOp>(producerOp)) ||
      !linalg::isElementwise(cast<linalg::LinalgOp>(consumerOp))) {
    return false;
  }
  return producerOp->hasOneUse();
}

struct ElementWiseFusion : ElementWiseFusionBase<ElementWiseFusion> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());

    linalg::ControlFusionFn fuseElementwiseOpsControlFn =
        [&](OpOperand *fusedOperand) {
          Operation *producer = fusedOperand->get().getDefiningOp();
          if (!producer)
            return false;
          Operation *consumer = fusedOperand->getOwner();
          return areFusableOps(producer, consumer);
        };

    linalg::populateElementwiseOpsFusionPatterns(patterns,
                                                 fuseElementwiseOpsControlFn);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // end namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createTileConsumerAndFuseProducersPass() {
  return std::make_unique<TileConsumerAndFuseProducers>();
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createElementWiseFusionPass() {
  return std::make_unique<ElementWiseFusion>();
}
