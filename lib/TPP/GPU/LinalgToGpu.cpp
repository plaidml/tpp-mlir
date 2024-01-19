//===- LinalgToGpu.cpp -------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"

#include "TPP/IR/MatcherUtils.h"
#include "TPP/Transforms/Utils/ValueUtils.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include <optional>

using namespace mlir;
using namespace mlir::tpp;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_LINALGTOGPU
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

// Creates an outermost parallel loop wrapper around an operation to represent
// number of GPU blocks.
// If there is already a parallel loop present, no operation is created and
// a nullopt is returned instead.
static std::optional<scf::ParallelOp>
createGpuBlocksWrapper(Operation *op, ArrayRef<int64_t> blockDims,
                       PatternRewriter &rewriter) {
  assert(blockDims.size() <= 3 && "Too many GPU blocks dimensions");

  auto loc = op->getLoc();

  auto *parentOp = op->getParentOp();
  if (isa<scf::ParallelOp>(parentOp))
    return std::nullopt;

  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

  SmallVector<Value> gpuBlocks;
  SmallVector<Value> lbs;
  SmallVector<Value> steps;
  for (auto blockDim : blockDims) {
    auto blockSize = rewriter.create<arith::ConstantIndexOp>(loc, blockDim);
    gpuBlocks.push_back(blockSize);
    // Add matching number of lbs and steps.
    lbs.push_back(zero);
    steps.push_back(one);
  }

  return rewriter.create<scf::ParallelOp>(loc, lbs, gpuBlocks, steps);
}

// Parse and return GPU computate capability.
static int getComputeCapability(llvm::StringRef chip) {
  llvm::StringRef delim = "_";
  llvm::StringRef ccToken =
      chip.substr(chip.find(delim) + delim.size(), chip.size());

  return std::stoi(ccToken.str());
}

// Return true if hardware supports WMMA operations.
static bool isMMASupported(LinalgToGpuOptions options) {
  return (options.gpuTriple == "nvptx64-nvidia-cuda") &&
         (getComputeCapability(options.gpuChip) >= 70);
}

// Helper struct containing hardware WMMA settings
// such as tile sizes.
struct WMMASettings {
  int m;
  int n;
  int k;
};

// Return true if the operation can be represented with WMMA compute.
static std::optional<WMMASettings> getWMMASettings(linalg::LinalgOp linalgOp,
                                                   LinalgToGpuOptions options,
                                                   int kTile) {
  if (!(isa<linalg::MatmulOp>(linalgOp) ||
        isa<linalg::BatchReduceMatmulOp>(linalgOp))) {
    return std::nullopt;
  }

  // Only static shapes are supported.
  if (linalgOp.hasDynamicShape())
    return std::nullopt;

  auto aType = linalgOp.getDpsInputs()[0].getType().cast<ShapedType>();
  auto bType = linalgOp.getDpsInputs()[1].getType().cast<ShapedType>();
  auto cType = linalgOp.getDpsInits()[0].getType().cast<ShapedType>();

  auto elemTypeA = aType.getElementType();
  auto elemTypeB = bType.getElementType();
  auto elemTypeC = cType.getElementType();

  // TODO: Add more WMMA combinations.
  bool isSupprtedPrecision =
      (elemTypeA.isF16() && elemTypeB.isF16() && elemTypeC.isF16()) ||
      (elemTypeA.isF16() && elemTypeB.isF16() && elemTypeC.isF32());
  if (!isSupprtedPrecision)
    return std::nullopt;

  auto mDim = cType.getShape()[0];
  auto nDim = cType.getShape()[1];
  auto kDim = aType.getShape().back();

  // Choose WMMA tile sizes.
  // The computation dimensions must fit into the tiles.
  // Reduction dimension tile size has to be compatible
  // with the hardware sizes.
  //
  // TODO: Add more possible tile sizes and choose optimal one.
  int wmmaTileM = 16;
  int wmmaTileN = 16;
  int wmmaTileK = 16;
  if ((mDim % wmmaTileM != 0) || (nDim % wmmaTileN != 0) ||
      (kDim % wmmaTileK != 0) || (kTile % wmmaTileK != 0)) {
    return std::nullopt;
  }

  return WMMASettings{wmmaTileM, wmmaTileN, wmmaTileK};
}

// Fuse a consumer using WMMA operations.
// Returns updated store op or nullopt if the fusion fails.
static std::optional<SmallVector<gpu::SubgroupMmaStoreMatrixOp>>
eltwiseFusion(linalg::LinalgOp rootOp, linalg::LinalgOp consumer,
              SmallVector<gpu::SubgroupMmaStoreMatrixOp> rootStoreOps,
              PatternRewriter &rewriter) {
  assert(rootStoreOps.size() > 0 && "Requires at least one store op");

  Location loc = rootOp.getLoc();

  auto rootOutput = rootOp.getDpsInits()[0];
  auto outputType = rootOutput.getType().cast<ShapedType>();

  // Must be a floating point type.
  // TODO: Add integer support.
  auto floatType = dyn_cast<FloatType>(outputType.getElementType());
  if (!floatType)
    return std::nullopt;

  // Insert fused eltwise ops before the store and later replace the store
  // with a new result.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(rootStoreOps[0]);

  // It is assumed that WMMA tile sizes do not vary between different
  // operations i.e., the original workload has been split into
  // a series of operations using the same WMMA configuration.
  gpu::MMAMatrixType mmaOutputType = rootStoreOps[0].getSrc().getType();
  auto leadingDim = rootStoreOps[0].getLeadDimension();

  // Collect new results after fusion.
  SmallVector<Value> fusedRes;

  SmallVector<Value> operands;
  if (structured_match::utils::isTwoDAddOp(consumer, &operands)) {
    // Get the value to be added - load the tile first.
    // Must be a buffer of the same type - scalar broadcast is not supported.
    // TODO: Consider adding support for eltwise with broadcast.
    auto addValue = (operands[0] != rootOutput) ? operands[0] : operands[1];
    if (addValue.getType() != rootOutput.getType())
      return std::nullopt;

    for (gpu::SubgroupMmaStoreMatrixOp rootStoreOp : rootStoreOps) {
      auto storeIndices = rootStoreOp.getIndices();

      // Fuse the add into the matmul body.
      auto loadOp =
          rewriter
              .create<gpu::SubgroupMmaLoadMatrixOp>(
                  loc, mmaOutputType, addValue, storeIndices, leadingDim,
                  /*transpose=*/UnitAttr())
              .getRes();
      auto eltwiseAttr = gpu::MMAElementwiseOp::ADDF;
      auto addRes =
          rewriter
              .create<gpu::SubgroupMmaElementwiseOp>(
                  loc, mmaOutputType, ValueRange{rootStoreOp.getSrc(), loadOp},
                  eltwiseAttr)
              .getRes();
      fusedRes.push_back(addRes);
    }
  } else if (structured_match::utils::isTwoDReluOp(consumer, &operands)) {
    Value zeroFloat = rewriter.create<arith::ConstantFloatOp>(
        loc, APFloat::getZero(floatType.getFloatSemantics()), floatType);

    Value zeroTile = rewriter.create<gpu::SubgroupMmaConstantMatrixOp>(
        loc, mmaOutputType, zeroFloat);
    for (auto rootStoreOp : rootStoreOps) {
      // Fuse the relu into the matmul body.
      auto eltwiseAttr = gpu::MMAElementwiseOp::MAXF;
      auto maxRes =
          rewriter
              .create<gpu::SubgroupMmaElementwiseOp>(
                  loc, mmaOutputType,
                  ValueRange{rootStoreOp.getSrc(), zeroTile}, eltwiseAttr)
              .getRes();
      fusedRes.push_back(maxRes);
    }
  } else {
    // Not a fusable operation. Bail out.
    return std::nullopt;
  }

  // Fusion must have failed, if number of new results is different.
  // Bail out.
  if (fusedRes.size() != rootStoreOps.size())
    return std::nullopt;

  // Store the new result.
  SmallVector<gpu::SubgroupMmaStoreMatrixOp> newStores;
  for (size_t i = 0; i < rootStoreOps.size(); i++) {
    auto storeIndices = rootStoreOps[i].getIndices();

    auto newStore = rewriter.create<gpu::SubgroupMmaStoreMatrixOp>(
        loc, fusedRes[i], rootStoreOps[i].getDstMemref(), storeIndices,
        leadingDim,
        /*transpose=*/UnitAttr());
    newStores.push_back(newStore);
  }

  // Replace store ops and cleanup standalone consumer.
  for (size_t i = 0; i < rootStoreOps.size(); i++)
    rewriter.replaceOp(rootStoreOps[i], newStores[i]);

  rewriter.eraseOp(consumer);

  return newStores;
}

// Fuse a consumer using scalar operations.
// TODO: Extend scalar fusion to support multiple stores.
//
// Returns updated store op or nullopt if the fusion fails.
static std::optional<memref::StoreOp> eltwiseFusion(linalg::LinalgOp rootOp,
                                                    linalg::LinalgOp consumer,
                                                    memref::StoreOp rootStoreOp,
                                                    PatternRewriter &rewriter) {
  Location loc = rootOp.getLoc();
  auto rootOutput = rootOp.getDpsInits()[0];
  auto outputType = rootOutput.getType().cast<ShapedType>();

  // Must be a floating point type.
  // TODO: Add integer support.
  auto floatType = dyn_cast<FloatType>(outputType.getElementType());
  if (!floatType)
    return std::nullopt;

  auto storeIndices = rootStoreOp.getIndices();

  // Insert fused eltwise ops before the store and later replace the store
  // with a new result.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(rootStoreOp);

  std::optional<memref::StoreOp> newStore = std::nullopt;
  SmallVector<Value> operands;
  if (structured_match::utils::isTwoDAddOp(consumer, &operands)) {
    // Get the value to be added. Load the element first, if necessary.
    auto addValue = (operands[0] != rootOutput) ? operands[0] : operands[1];
    if (addValue.getType().isa<ShapedType>()) {
      addValue = rewriter.create<memref::LoadOp>(loc, addValue, storeIndices)
                     .getResult();
    }
    // Fuse the add into the matmul body.
    auto addOp =
        rewriter.create<arith::AddFOp>(loc, rootStoreOp.getValue(), addValue);
    // Store the new result.
    newStore = rewriter.replaceOpWithNewOp<memref::StoreOp>(
        rootStoreOp, addOp.getResult(), rootOutput, storeIndices);
  } else if (structured_match::utils::isTwoDReluOp(consumer, &operands)) {
    // Fuse the relu into the matmul body.
    Value zeroFloat = rewriter.create<arith::ConstantFloatOp>(
        loc, APFloat::getZero(floatType.getFloatSemantics()), floatType);
    auto maxOp = rewriter.create<arith::MaximumFOp>(loc, rootStoreOp.getValue(),
                                                    zeroFloat);
    // Store the new result.
    newStore = rewriter.replaceOpWithNewOp<memref::StoreOp>(
        rootStoreOp, maxOp.getResult(), rootOutput, storeIndices);
  } else {
    // Not a fusable operation. Bail out.
    return std::nullopt;
  }

  rewriter.eraseOp(consumer);

  return newStore;
}

// Find operations fusable with the given root op.
//
// A simple fusion strategy that looks at the other operations after the root
// linalg op and tries to fuse them.
static SmallVector<linalg::LinalgOp>
getFusableConsumers(linalg::LinalgOp rootOp) {
  auto *parentOp = rootOp->getParentOp();
  auto rootOutput = rootOp.getDpsInits()[0];

  // Traverse other ops within the same region and collect consumers.
  SmallVector<linalg::LinalgOp> consumers;
  Operation *nextOp = rootOp;
  while ((nextOp = nextOp->getNextNode())) {
    // Potential consumers must be within the same region.
    if (nextOp->getParentOp() != parentOp)
      break;

    // Only other linalg ops are expected as consumers.
    // TODO: might need to be relaxed to skip over ops without side effects
    auto consumer = dyn_cast<linalg::LinalgOp>(nextOp);
    if (!consumer || !linalg::isElementwise(consumer))
      break;
    // Require the same iteration space.
    if (consumer.getNumParallelLoops() != rootOp.getNumParallelLoops())
      break;

    auto outBuf = consumer.getDpsInitOperand(0)->get();
    // Check that the op reuses the same output buffer as the root op.
    // Otherwise, it is assumed that the op cannot be fused.
    // TODO: Consider adding support for eltwise with broadcast.
    if (outBuf != rootOutput)
      break;

    consumers.push_back(consumer);
  }

  return consumers;
}

// Fuse elementwise consumers within a GPU kernel.
//
// Fusion bails on the first mismatch.
// Returns updated store ops.
template <typename StoreTy>
static StoreTy fuseEltwiseConsumers(linalg::LinalgOp rootOp,
                                    StoreTy rootStoreOps,
                                    PatternRewriter &rewriter) {
  // Constrain conversion to the supported fusion types.
  static_assert(
      llvm::is_one_of<StoreTy, memref::StoreOp,
                      SmallVector<gpu::SubgroupMmaStoreMatrixOp>>::value);

  auto consumers = getFusableConsumers(rootOp);

  for (auto op : consumers) {
    std::optional<StoreTy> updatedStoreOps = std::nullopt;

    updatedStoreOps = eltwiseFusion(rootOp, op, rootStoreOps, rewriter);

    // Failed to fuse operation. Bail out.
    if (!updatedStoreOps)
      break;

    rootStoreOps = *updatedStoreOps;
  }

  return rootStoreOps;
}

// Create WMMA instructions out of matmul-like operation.
static LogicalResult gemmToGpuMMA(linalg::LinalgOp linalgOp,
                                  WMMASettings wmmaSettings, int kTile,
                                  PatternRewriter &rewriter) {
  assert((isa<linalg::MatmulOp>(linalgOp) ||
          isa<linalg::BatchReduceMatmulOp>(linalgOp)) &&
         "Requires a matmul like op for MMA lowering");

  Location loc = linalgOp.getLoc();

  // If there is no parallel loop, create a unit blocks wrapper around the
  // current op.
  // This ensures that WMMA operations are created at the thread level (inner
  // nested parallel loops).
  auto blocksLoop = createGpuBlocksWrapper(linalgOp, {1, 1}, rewriter);
  if (blocksLoop)
    rewriter.setInsertionPoint(blocksLoop->getBody()->getTerminator());

  auto matA = linalgOp.getDpsInputs()[0];
  auto matB = linalgOp.getDpsInputs()[1];
  auto matC = linalgOp.getDpsInits()[0];

  auto typeA = matA.getType().cast<ShapedType>();
  auto typeB = matB.getType().cast<ShapedType>();
  auto typeC = matC.getType().cast<ShapedType>();

  auto stridesA = utils::getStaticStrides(matA);
  auto stridesB = utils::getStaticStrides(matB);
  auto stridesC = utils::getStaticStrides(matC);

  if (failed(stridesA) || failed(stridesB) || failed(stridesC)) {
    return rewriter.notifyMatchFailure(
        linalgOp, "Expect static strides for MMA lowering");
  }
  if (stridesA->back() != 1 || stridesB->back() != 1 || stridesC->back() != 1) {
    return rewriter.notifyMatchFailure(
        linalgOp,
        "Expect unit stride in the innermost dimension for MMA operations");
  }

  int dimM = typeC.getShape()[0];
  int dimN = typeC.getShape()[1];
  int dimK = typeA.getShape().back();

  gpu::MMAMatrixType mmaTypeA = gpu::MMAMatrixType::get(
      {wmmaSettings.m, wmmaSettings.k}, typeA.getElementType(), "AOp");
  gpu::MMAMatrixType mmaTypeB = gpu::MMAMatrixType::get(
      {wmmaSettings.k, wmmaSettings.n}, typeB.getElementType(), "BOp");
  gpu::MMAMatrixType mmaTypeC = gpu::MMAMatrixType::get(
      {wmmaSettings.m, wmmaSettings.n}, typeC.getElementType(), "COp");

  bool isBrgemm = isa<linalg::BatchReduceMatmulOp>(linalgOp);

  // Skip batch dimension stride in case of brgemm.
  auto lda = rewriter.getIndexAttr(stridesA->begin()[isBrgemm ? 1 : 0]);
  auto ldb = rewriter.getIndexAttr(stridesB->begin()[isBrgemm ? 1 : 0]);
  auto ldc = rewriter.getIndexAttr(stridesC->front());

  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  // WMMA requires warp/subgroup size of 32 threads/work items.
  Value subgroupSize = rewriter.create<arith::ConstantIndexOp>(loc, 32);

  // Create parallel loop to indicate that the whole subgroup is performing MMA
  // operations together. It also ensures that the kernel is outlined with
  // the correct number of threads.
  auto parallelLoop = rewriter.create<scf::ParallelOp>(
      loc, ValueRange{zero}, ValueRange{subgroupSize}, ValueRange{one});

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(parallelLoop.getBody()->getTerminator());

  // Fetch the inital value of the output element.
  SmallVector<Value> tilesC;
  for (int m = 0; m < dimM; m += wmmaSettings.m) {
    for (int n = 0; n < dimN; n += wmmaSettings.n) {
      Value rowIdx = rewriter.create<arith::ConstantIndexOp>(loc, m);
      Value colIdx = rewriter.create<arith::ConstantIndexOp>(loc, n);
      Value tileC =
          rewriter
              .create<gpu::SubgroupMmaLoadMatrixOp>(
                  loc, mmaTypeC, matC, ValueRange{rowIdx, colIdx}, ldc,
                  /*transpose=*/UnitAttr())
              .getRes();
      tilesC.push_back(tileC);
    }
  }

  // Create a loop and step into it.
  auto startLoop = [&](int lb, int ub, int step) -> scf::ForOp {
    Value lbCst = rewriter.create<arith::ConstantIndexOp>(loc, lb);
    Value ubCst = rewriter.create<arith::ConstantIndexOp>(loc, ub);
    Value stepCst = rewriter.create<arith::ConstantIndexOp>(loc, step);
    scf::ForOp loopOp =
        rewriter.create<scf::ForOp>(loc, lbCst, ubCst, stepCst, tilesC);
    rewriter.setInsertionPointToStart(loopOp.getBody());
    return loopOp;
  };
  auto getLoopIterValues = [&](scf::ForOp loopOp) -> SmallVector<Value> {
    SmallVector<Value> loopIterVals;
    for (auto iterArg : loopOp.getRegionIterArgs())
      loopIterVals.push_back(iterArg);
    return loopIterVals;
  };

  // Construct and move into batch reduction loop.
  // Propagate output values as iter args.
  scf::ForOp batchLoop;
  Value batchIv;
  if (isBrgemm) {
    batchLoop = startLoop(0, typeA.getShape()[0], 1);
    batchIv = batchLoop.getInductionVar();
    tilesC = getLoopIterValues(batchLoop);
  }

  // Construct and move into GEMM reduction dimension tiling loop.
  // Propagate output values as iter args.
  scf::ForOp kDimLoop = startLoop(0, dimK, kTile);
  Value kDimIv = kDimLoop.getInductionVar();
  tilesC = getLoopIterValues(kDimLoop);

  // Load A sub-tiles.
  SmallVector<Value> tilesA;
  for (int m = 0; m < dimM; m += wmmaSettings.m) {
    for (int k = 0; k < kTile; k += wmmaSettings.k) {
      Value rowOffset = rewriter.create<arith::ConstantIndexOp>(loc, m);
      Value colOffset = rewriter.create<arith::ConstantIndexOp>(loc, k);

      Value rowIdx = rowOffset;
      Value colIdx = rewriter.create<arith::AddIOp>(loc, kDimIv, colOffset);

      Value tileA = rewriter
                        .create<gpu::SubgroupMmaLoadMatrixOp>(
                            loc, mmaTypeA, matA,
                            isBrgemm ? ValueRange{batchIv, rowIdx, colIdx}
                                     : ValueRange{rowIdx, colIdx},
                            lda,
                            /*transpose=*/UnitAttr())
                        .getRes();
      tilesA.push_back(tileA);
    }
  }

  // Load B sub-tiles.
  SmallVector<Value> tilesB;
  for (int k = 0; k < kTile; k += wmmaSettings.k) {
    for (int n = 0; n < dimN; n += wmmaSettings.n) {
      Value rowOffset = rewriter.create<arith::ConstantIndexOp>(loc, k);
      Value colOffset = rewriter.create<arith::ConstantIndexOp>(loc, n);

      Value rowIdx = rewriter.create<arith::AddIOp>(loc, kDimIv, rowOffset);
      Value colIdx = colOffset;

      Value tileB = rewriter
                        .create<gpu::SubgroupMmaLoadMatrixOp>(
                            loc, mmaTypeB, matB,
                            isBrgemm ? ValueRange{batchIv, rowIdx, colIdx}
                                     : ValueRange{rowIdx, colIdx},
                            ldb, /*transpose=*/UnitAttr())
                        .getRes();
      tilesB.push_back(tileB);
    }
  }

  const int numTilesM = dimM / wmmaSettings.m;
  const int numTilesN = dimN / wmmaSettings.n;
  const int numTilesK = kTile / wmmaSettings.k;

  // Compute sub-tiles of the C tile.
  //
  // Iterate over the reduction dimension sub-tiles as the outermost
  // loop to minimize read after write conflicts between partial
  // computations of the same C sub-tile.
  //
  // Initialize sub-tiles with the loaded C tiles.
  SmallVector<Value> results = tilesC;
  for (int k = 0; k < numTilesK; k++) {
    for (int m = 0; m < numTilesM; m++) {
      for (int n = 0; n < numTilesN; n++) {
        int aIdx = m * numTilesK + k;
        int bIdx = k * numTilesN + n;
        int cIdx = m * numTilesN + n;

        Value result = rewriter
                           .create<gpu::SubgroupMmaComputeOp>(
                               loc, tilesC[cIdx].getType(), tilesA[aIdx],
                               tilesB[bIdx], results[cIdx],
                               /*a_transpose=*/UnitAttr(),
                               /*b_transpose=*/UnitAttr())
                           .getRes();
        // Update sub-tile partial result.
        results[cIdx] = result;
      }
    }
  }

  // Create loop terminator and exit the loop.
  auto terminateLoop = [&](scf::ForOp loopOp, SmallVector<Value> resultValues) {
    rewriter.setInsertionPointToEnd(loopOp.getBody());
    rewriter.create<scf::YieldOp>(loc, resultValues);
    rewriter.setInsertionPointAfter(loopOp);
  };

  // Terminate and exit reduction dim loop.
  terminateLoop(kDimLoop, results);
  results = kDimLoop.getResults();

  // Terminate and exit batch reduce loop.
  if (isBrgemm) {
    terminateLoop(batchLoop, results);
    results = batchLoop.getResults();
  }

  // Write back the final C sub-tiles results to the output buffer.
  SmallVector<gpu::SubgroupMmaStoreMatrixOp> storeOps;
  for (int m = 0; m < numTilesM; m++) {
    for (int n = 0; n < numTilesN; n++) {
      int resIdx = m * numTilesN + n;

      Value rowIdx =
          rewriter.create<arith::ConstantIndexOp>(loc, m * wmmaSettings.m);
      Value colIdx =
          rewriter.create<arith::ConstantIndexOp>(loc, n * wmmaSettings.n);
      auto storeOp = rewriter.create<gpu::SubgroupMmaStoreMatrixOp>(
          loc, results[resIdx], matC, ValueRange{rowIdx, colIdx}, ldc,
          /*transpose=*/UnitAttr());
      storeOps.push_back(storeOp);
    }
  }

  (void)fuseEltwiseConsumers<SmallVector<gpu::SubgroupMmaStoreMatrixOp>>(
      linalgOp, storeOps, rewriter);

  rewriter.eraseOp(linalgOp);

  return success();
}

// Create loops out of matmul-like operation.
static LogicalResult gemmToGpuLoops(linalg::LinalgOp linalgOp,
                                    PatternRewriter &rewriter) {
  assert((isa<linalg::MatmulOp>(linalgOp) ||
          isa<linalg::BatchReduceMatmulOp>(linalgOp)) &&
         "Requires a matmul like op for loop lowering");

  Location loc = linalgOp.getLoc();

  auto matA = linalgOp.getDpsInputs()[0];
  auto matB = linalgOp.getDpsInputs()[1];
  auto matC = linalgOp.getDpsInits()[0];

  ArrayRef<int64_t> shapeC = matC.getType().cast<ShapedType>().getShape();
  ArrayRef<int64_t> shapeA = matA.getType().cast<ShapedType>().getShape();

  // Parallel dims.
  Value i = rewriter.create<arith::ConstantIndexOp>(loc, shapeC[0]);
  Value j = rewriter.create<arith::ConstantIndexOp>(loc, shapeC[1]);
  // Reduction dim.
  Value k = rewriter.create<arith::ConstantIndexOp>(loc, shapeA.back());
  // Lbs.
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  // Step.
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Value> ivs;

  // Create parallel loops over the outer dimensions.
  auto parallelLoop = rewriter.create<scf::ParallelOp>(
      loc, ValueRange{zero, zero}, ValueRange{i, j}, ValueRange{one, one});
  auto parallelIvs = parallelLoop.getInductionVars();
  ivs.append(parallelIvs.begin(), parallelIvs.end());

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(parallelLoop.getBody()->getTerminator());

  // Fetch the inital value of the output element.
  Value initVal =
      rewriter.create<memref::LoadOp>(loc, matC, parallelIvs).getResult();

  bool isBrgemm = isa<linalg::BatchReduceMatmulOp>(linalgOp);
  scf::ForOp batchLoop;
  Value batchIv;
  if (isBrgemm) {
    Value batch = rewriter.create<arith::ConstantIndexOp>(loc, shapeA[0]);
    batchLoop =
        rewriter.create<scf::ForOp>(loc, zero, batch, one, ValueRange{initVal});
    rewriter.setInsertionPointToStart(batchLoop.getBody());
    batchIv = batchLoop.getInductionVar();
    initVal = batchLoop.getRegionIterArg(0);
  }

  // Compute matmul with a loop over reduction dimension.
  // Each GPU thread computes a single result element.
  // Accumulate result locally through loop's iter args.
  // This maps to more efficient computation as the accumulation is kept
  // locally by a thread.
  auto bodyBuilder = [&](OpBuilder &b, Location loc, Value localIv,
                         ValueRange iterArgs) {
    SmallVector<Value> loopIvs = ivs;
    loopIvs.push_back(localIv);
    assert(loopIvs.size() == 3);
    Value localI = loopIvs[0];
    Value localJ = loopIvs[1];
    Value localK = loopIvs[2];
    Value scalarA =
        b.create<memref::LoadOp>(loc, matA,
                                 isBrgemm ? ValueRange{batchIv, localI, localK}
                                          : ValueRange{localI, localK});
    Value scalarB =
        b.create<memref::LoadOp>(loc, matB,
                                 isBrgemm ? ValueRange{batchIv, localK, localJ}
                                          : ValueRange{localK, localJ});
    Value scalarMul = b.create<arith::MulFOp>(loc, scalarA, scalarB);
    auto scalarAdd = b.create<arith::AddFOp>(loc, iterArgs[0], scalarMul);

    b.create<scf::YieldOp>(loc, scalarAdd.getResult());
  };
  auto accumulationLoop = rewriter.create<scf::ForOp>(
      loc, zero, k, one, ValueRange{initVal},
      [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
        bodyBuilder(b, loc, iv, iterArgs);
      });

  Value result = accumulationLoop.getResults()[0];

  if (isBrgemm) {
    rewriter.setInsertionPointToEnd(batchLoop.getBody());
    rewriter.create<scf::YieldOp>(loc, ValueRange{result});
    result = batchLoop.getResults()[0];
    rewriter.setInsertionPointAfter(batchLoop);
  }

  // Write back the total sum to the output buffer.
  auto storeOp =
      rewriter.create<memref::StoreOp>(loc, result, matC, parallelIvs);

  (void)fuseEltwiseConsumers<memref::StoreOp>(linalgOp, storeOp, rewriter);

  rewriter.eraseOp(linalgOp);

  return success();
}

// Convert linalg.matmul or linalg.batch_reduce_matmul to GPU-compatible kernel.
template <typename LinalgOpTy>
struct ConvertGemmLikeToGpu : public OpRewritePattern<LinalgOpTy> {
  using OpRewritePattern<LinalgOpTy>::OpRewritePattern;
  // Constrain conversion to the supported GEMM-like ops.
  static_assert(llvm::is_one_of<LinalgOpTy, linalg::MatmulOp,
                                linalg::BatchReduceMatmulOp>::value);

  ConvertGemmLikeToGpu(MLIRContext *ctx, LinalgToGpuOptions options)
      : OpRewritePattern<LinalgOpTy>(ctx), options(options) {}

  LogicalResult matchAndRewrite(LinalgOpTy gemmLikeOp,
                                PatternRewriter &rewriter) const override {
    if (!gemmLikeOp.hasPureBufferSemantics()) {
      return rewriter.notifyMatchFailure(
          gemmLikeOp, "Linalg brgemm to GPU expects memref type");
    }
    if (gemmLikeOp.hasDynamicShape()) {
      return rewriter.notifyMatchFailure(
          gemmLikeOp, "Expect static shape when mapping to GPU");
    }

    auto aType =
        gemmLikeOp.getDpsInputs()[0].getType().template cast<ShapedType>();
    auto kDim = aType.getShape().back();
    auto kTile = kDim < options.kTile ? kDim : options.kTile;

    if (options.useWmma && isMMASupported(options)) {
      if (auto settings = getWMMASettings(gemmLikeOp, options, kTile))
        return gemmToGpuMMA(gemmLikeOp, *settings, kTile, rewriter);
    }
    return gemmToGpuLoops(gemmLikeOp, rewriter);
  }

private:
  LinalgToGpuOptions options;
};

void populateLinalgToGpuPatterns(RewritePatternSet &patterns,
                                 LinalgToGpuOptions options) {
  patterns.add<ConvertGemmLikeToGpu<linalg::MatmulOp>,
               ConvertGemmLikeToGpu<linalg::BatchReduceMatmulOp>>(
      patterns.getContext(), options);
}

struct LinalgToGpu : public tpp::impl::LinalgToGpuBase<LinalgToGpu> {
  using LinalgToGpuBase::LinalgToGpuBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateLinalgToGpuPatterns(
        patterns,
        LinalgToGpuOptions{useWmma, gpuTriple, gpuChip, gpuFeatures, kTile});
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace
