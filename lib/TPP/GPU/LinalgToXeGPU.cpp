//===- LinalgToXeGPU.cpp -----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"

#include "TPP/Dialect/XeGPU/IR/XeGPUOps.h"
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
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include <optional>

using namespace mlir;
using namespace mlir::tpp;
using namespace imex;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_LINALGTOXEGPU
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

// Return DPAS tile sizes if the gemm-like operation fits DPAS hardware.
static std::optional<SmallVector<int64_t>>
getDPASConfig(linalg::LinalgOp linalgOp, int kTile) {
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

  // TODO: Add more DPAS combinations.
  bool isSupportedPrecision =
      (elemTypeA.isF16() && elemTypeB.isF16() && elemTypeC.isF16()) ||
      (elemTypeA.isF16() && elemTypeB.isF16() && elemTypeC.isF32());
  if (!isSupportedPrecision)
    return std::nullopt;

  auto mDim = cType.getShape()[0];
  auto nDim = cType.getShape()[1];
  auto kDim = aType.getShape().back();

  // DPAS hardware sizes in MxNxK format.
  // TODO: In case more hardware configurations are available,
  //       add some automatic selection for optimal sizes.
  SmallVector<int64_t> dpasTile{8, 16, 16};

  // Validate workload sizes.
  // The computation dimensions must fit into the tiles.
  // Reduction dimension tile size has to be compatible
  // with the warp tile.
  int dpasTileM = dpasTile[0];
  int dpasTileN = dpasTile[1];
  int dpasTileK = dpasTile[2];
  if ((mDim % dpasTileM != 0) || (nDim % dpasTileN != 0) ||
      (kDim % dpasTileK != 0) || (kTile % dpasTileK != 0)) {
    return std::nullopt;
  }

  return dpasTile;
}

// Fuse an elementwise consumer operation.
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
    // TODO: Add support for eltwise with broadcast.
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

// Create DPAS instructions out of GEMM-like operation.
static LogicalResult gemmToDPAS(linalg::LinalgOp linalgOp,
                                ArrayRef<int64_t> dpasTile, int kTile,
                                PatternRewriter &rewriter) {
  assert((isa<linalg::MatmulOp>(linalgOp) ||
          isa<linalg::BatchReduceMatmulOp>(linalgOp)) &&
         "Requires a GEMM-like op for DPAS lowering");

  Location loc = linalgOp.getLoc();
  auto ctx = linalgOp.getContext();

  // If there is no parallel loop, create a unit blocks wrapper around the
  // current op.
  // This ensures that WMMA operations are created at the thread level (inner
  // nested parallel loops).
  //
  // TODO: Move out to a separate pass before linalg lowering.
  //       At this point GPU operations should already be outlined.
  // auto blocksLoop = createGpuBlocksWrapper(linalgOp, {1, 1}, rewriter);
  // if (blocksLoop)
  //   rewriter.setInsertionPoint(blocksLoop->getBody()->getTerminator());

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

  int64_t dpasTileM = dpasTile[0];
  int64_t dpasTileN = dpasTile[1];
  int64_t dpasTileK = dpasTile[2];

  // Tensor descriptors.
  auto dpasTypeA = xegpu::TensorDescType::get({dpasTileM, dpasTileK},
                                              typeA.getElementType());
  auto dpasTypeB = xegpu::TensorDescType::get({dpasTileK, dpasTileN},
                                              typeB.getElementType());
  auto dpasTypeC = xegpu::TensorDescType::get({dpasTileM, dpasTileN},
                                              typeC.getElementType());

  // Instruction mode - use workgroup intristic directly.
  auto xegpuMode = xegpu::Mode::VC;

  // Cache hints for loads and stores.
  auto readCacheHint =
      xegpu::CacheReadHintAttr::get(ctx, xegpu::CacheReadHint::CACHED);
  auto writeCacheHint =
      xegpu::CacheWriteHintAttr::get(ctx, xegpu::CacheWriteHint::WRITE_BACK);

  // No transposition needed - argument only required for the build API.
  DenseI64ArrayAttr transpose = nullptr;

  bool isBrgemm = isa<linalg::BatchReduceMatmulOp>(linalgOp);

  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

  int dimM = typeC.getShape()[0];
  int dimN = typeC.getShape()[1];
  int dimK = typeA.getShape().back();

  // Create C sub-tiles.
  SmallVector<Value> tilesC;
  for (int m = 0; m < dimM; m += dpasTileM) {
    for (int n = 0; n < dimN; n += dpasTileN) {
      Value rowIdx = rewriter.create<arith::ConstantIndexOp>(loc, m);
      Value colIdx = rewriter.create<arith::ConstantIndexOp>(loc, n);

      mlir::SmallVector<mlir::OpFoldResult> loadOffsets{rowIdx, colIdx};
      Value tileC =
          rewriter
              .create<xegpu::CreateNdDescOp>(loc, dpasTypeC, matC, loadOffsets,
                                             /*boundary_check=*/true, xegpuMode)
              .getResult();
      tilesC.push_back(tileC);
    }
  }

  // Fetch the inital value of the output element.
  auto vecTypeC =
      VectorType::get(dpasTypeC.getShape(), dpasTypeC.getElementType());

  SmallVector<Value> loadVecC;
  for (auto tileC : tilesC) {
    auto loadOp = rewriter.create<xegpu::LoadNDOp>(
        loc, vecTypeC, tileC, /*vnni_axis=*/nullptr, transpose,
        /*l1_hint=*/readCacheHint,
        /*l2_hint=*/readCacheHint, /*l3_hint=*/readCacheHint, xegpuMode);
    loadVecC.push_back(loadOp);
  }
  rewriter.create<xegpu::CompileHintOp>(loc);

  // DPAS only works with F32 accumulators.
  auto dpasResType =
      VectorType::get(dpasTypeC.getShape(), FloatType::getF32(ctx));

  // Extend the accumulation values if needed.
  auto isOutF16 = typeC.getElementType().isF16();
  if (isOutF16) {
    for (size_t i = 0; i < loadVecC.size(); i++) {
      auto extOp =
          rewriter.create<arith::ExtFOp>(loc, dpasResType, loadVecC[i]);
      loadVecC[i] = extOp.getOut();
    }
  }

  // Create a loop and step into it.
  auto startLoop = [&](int lb, int ub, int step,
                       ValueRange iterArgs) -> scf::ForOp {
    Value lbCst = rewriter.create<arith::ConstantIndexOp>(loc, lb);
    Value ubCst = rewriter.create<arith::ConstantIndexOp>(loc, ub);
    Value stepCst = rewriter.create<arith::ConstantIndexOp>(loc, step);
    scf::ForOp loopOp =
        rewriter.create<scf::ForOp>(loc, lbCst, ubCst, stepCst, iterArgs);
    rewriter.setInsertionPointToStart(loopOp.getBody());
    return loopOp;
  };
  auto getLoopIterValues = [&](scf::ForOp loopOp) -> SmallVector<Value> {
    SmallVector<Value> loopIterVals;
    for (auto iterArg : loopOp.getRegionIterArgs())
      loopIterVals.push_back(iterArg);
    return loopIterVals;
  };

  OpBuilder::InsertionGuard guard(rewriter);

  // Construct and move into batch reduction loop.
  // Propagate output values as iter args.
  scf::ForOp batchLoop;
  Value batchIv;
  if (isBrgemm) {
    batchLoop = startLoop(0, typeA.getShape()[0], 1, loadVecC);
    batchIv = batchLoop.getInductionVar();
    loadVecC = getLoopIterValues(batchLoop);
  }

  // Create A sub-tiles.
  SmallVector<Value> tilesA;
  for (int m = 0; m < dimM; m += dpasTileM) {
    for (int k = 0; k < kTile; k += dpasTileK) {
      Value rowIdx = rewriter.create<arith::ConstantIndexOp>(loc, m);
      Value colIdx = rewriter.create<arith::ConstantIndexOp>(loc, k);

      mlir::SmallVector<mlir::OpFoldResult> loadOffsets;
      if (isBrgemm)
        loadOffsets.push_back(batchIv);
      loadOffsets.append({rowIdx, colIdx});
      Value tileA =
          rewriter
              .create<xegpu::CreateNdDescOp>(loc, dpasTypeA, matA, loadOffsets,
                                             /*boundary_check=*/true, xegpuMode)
              .getResult();
      tilesA.push_back(tileA);
    }
  }

  // Create B sub-tiles.
  SmallVector<Value> tilesB;
  for (int k = 0; k < kTile; k += dpasTileK) {
    for (int n = 0; n < dimN; n += dpasTileN) {
      Value rowIdx = rewriter.create<arith::ConstantIndexOp>(loc, k);
      Value colIdx = rewriter.create<arith::ConstantIndexOp>(loc, n);

      mlir::SmallVector<mlir::OpFoldResult> loadOffsets;
      if (isBrgemm)
        loadOffsets.push_back(batchIv);
      loadOffsets.append({rowIdx, colIdx});
      Value tileB =
          rewriter
              .create<xegpu::CreateNdDescOp>(loc, dpasTypeB, matB, loadOffsets,
                                             /*boundary_check=*/true, xegpuMode)
              .getResult();
      tilesB.push_back(tileB);
    }
  }

  // Construct and move into GEMM reduction dimension tiling loop.
  // Propagate output values as iter args.
  SmallVector<Value> iterArgs;
  iterArgs.append(loadVecC);
  iterArgs.append(tilesA);
  iterArgs.append(tilesB);
  scf::ForOp kDimLoop = startLoop(0, dimK, kTile, iterArgs);
  auto iterValues = getLoopIterValues(kDimLoop);

  loadVecC = SmallVector<Value>{iterValues.begin(),
                                iterValues.begin() + loadVecC.size()};
  tilesA =
      SmallVector<Value>{iterValues.begin() + loadVecC.size(),
                         iterValues.begin() + loadVecC.size() + tilesA.size()};
  tilesB =
      SmallVector<Value>{iterValues.end() - tilesB.size(), iterValues.end()};

  // TODO: Make the VNNI factor a flexible parameter.
  const int vnniFactor = 2;
  auto getVnniVector = [&](ArrayRef<int64_t> shape, Type elementType,
                           int vnniAxis) -> VectorType {
    SmallVector<int64_t> vecShape{shape};
    vecShape[vnniAxis] /= vnniFactor;
    vecShape.push_back(vnniFactor);
    return VectorType::get(vecShape, elementType);
  };

  // Load A sub-tiles.
  auto vnniAxisAttrA = IntegerAttr::get(rewriter.getI32Type(), 1);
  auto vecTypeA = getVnniVector(
      dpasTypeA.getShape(), dpasTypeA.getElementType(), vnniAxisAttrA.getInt());

  SmallVector<Value> loadVecA;
  for (auto tileA : tilesA) {
    auto loadOp = rewriter.create<xegpu::LoadNDOp>(
        loc, vecTypeA, tileA, vnniAxisAttrA, transpose,
        /*l1_hint=*/readCacheHint,
        /*l2_hint=*/readCacheHint, /*l3_hint=*/readCacheHint, xegpuMode);
    loadVecA.push_back(loadOp);
  }

  // Load B sub-tiles.
  auto vnniAxisAttrB = IntegerAttr::get(rewriter.getI32Type(), 0);
  auto vecTypeB = getVnniVector(
      dpasTypeB.getShape(), dpasTypeB.getElementType(), vnniAxisAttrB.getInt());

  SmallVector<Value> loadVecB;
  for (auto tileB : tilesB) {
    auto loadOp = rewriter.create<xegpu::LoadNDOp>(
        loc, vecTypeB, tileB, vnniAxisAttrB, transpose,
        /*l1_hint=*/readCacheHint,
        /*l2_hint=*/readCacheHint, /*l3_hint=*/readCacheHint, xegpuMode);
    loadVecB.push_back(loadOp);
  }

  // Update offsets of the input tiles.
  // Shift along the reduction dimension.
  Value kTileOffset = rewriter.create<arith::ConstantIndexOp>(loc, kTile);
  for (size_t i = 0; i < tilesA.size(); i++) {
    auto updatedTile = rewriter
                           .create<xegpu::UpdateNDOffsetOp>(
                               loc, dpasTypeA, tilesA[i],
                               ValueRange{zero, kTileOffset}, xegpuMode)
                           .getResult();
    tilesA[i] = updatedTile;
  }
  for (size_t i = 0; i < tilesB.size(); i++) {
    auto updatedTile = rewriter
                           .create<xegpu::UpdateNDOffsetOp>(
                               loc, dpasTypeB, tilesB[i],
                               ValueRange{kTileOffset, zero}, xegpuMode)
                           .getResult();
    tilesB[i] = updatedTile;
  }

  // Prefetch the next set of input tiles.
  for (auto tileA : tilesA) {
    rewriter.create<xegpu::PrefetchNDOp>(loc, tileA, /*l1_hint=*/readCacheHint,
                                         /*l2_hint=*/readCacheHint,
                                         /*l3_hint=*/readCacheHint, xegpuMode);
  }
  for (auto tileB : tilesB) {
    rewriter.create<xegpu::PrefetchNDOp>(loc, tileB, /*l1_hint=*/readCacheHint,
                                         /*l2_hint=*/readCacheHint,
                                         /*l3_hint=*/readCacheHint, xegpuMode);
  }
  // Ensure that prefetches are scheduled before computation starts.
  rewriter.create<xegpu::CompileHintOp>(loc);

  const int numTilesM = dimM / dpasTileM;
  const int numTilesN = dimN / dpasTileN;
  const int numTilesK = kTile / dpasTileK;

  // Compute sub-tiles of the C tile.
  //
  // Iterate over the reduction dimension sub-tiles as the outermost
  // loop to minimize read after write conflicts between partial
  // computations of the same C sub-tile.
  SmallVector<Value> dpasResults = loadVecC;

  for (int k = 0; k < numTilesK; k++) {
    for (int m = 0; m < numTilesM; m++) {
      for (int n = 0; n < numTilesN; n++) {
        int aIdx = m * numTilesK + k;
        int bIdx = k * numTilesN + n;
        int cIdx = m * numTilesN + n;

        Value result = rewriter
                           .create<xegpu::DpasOp>(
                               loc, dpasResType, loadVecA[aIdx], loadVecB[bIdx],
                               dpasResults[cIdx], xegpuMode)
                           .getResult();

        // Update sub-tile partial result.
        dpasResults[cIdx] = result;
      }
    }
  }

  // Ensure that DPAS computation is finished before the input tiles are
  // replaced with new values.
  rewriter.create<xegpu::CompileHintOp>(loc);
  rewriter.create<gpu::BarrierOp>(loc);

  // Create loop terminator and exit the loop.
  auto terminateLoop = [&](scf::ForOp loopOp, SmallVector<Value> resultValues) {
    rewriter.setInsertionPointToEnd(loopOp.getBody());
    rewriter.create<scf::YieldOp>(loc, resultValues);
    rewriter.setInsertionPointAfter(loopOp);
  };

  SmallVector<Value> yieldVals;
  yieldVals.append(dpasResults);
  yieldVals.append(tilesA);
  yieldVals.append(tilesB);

  // Terminate and exit reduction dim loop.
  terminateLoop(kDimLoop, yieldVals);
  yieldVals = kDimLoop.getResults();

  SmallVector<Value> results{yieldVals.begin(),
                             yieldVals.begin() + dpasResults.size()};

  // Terminate and exit batch reduce loop.
  if (isBrgemm) {
    terminateLoop(batchLoop, results);
    results = batchLoop.getResults();
  }

  // Truncate the result values if needed.
  if (isOutF16) {
    auto truncType =
        VectorType::get(dpasTypeC.getShape(), FloatType::getF16(ctx));
    for (size_t i = 0; i < results.size(); i++) {
      auto truncOp =
          rewriter.create<arith::TruncFOp>(loc, truncType, results[i]);
      results[i] = truncOp.getOut();
    }
  }

  // Write back the final C sub-tiles results to the output buffer.
  SmallVector<xegpu::StoreNDOp> storeOps;
  for (size_t i = 0; i < tilesC.size(); i++) {
    auto storeOp = rewriter.create<xegpu::StoreNDOp>(loc, tilesC[i], results[i],
                                                     /*l1_hint=*/writeCacheHint,
                                                     /*l2_hint=*/writeCacheHint,
                                                     /*l3_hint=*/writeCacheHint,
                                                     xegpuMode);
    storeOps.push_back(storeOp);
  }

  // (void)fuseEltwiseConsumers<SmallVector<gpu::SubgroupMmaStoreMatrixOp>>(
  //     linalgOp, storeOps, rewriter);

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

  ConvertGemmLikeToGpu(MLIRContext *ctx, LinalgToXeGPUOptions options)
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

    // Ensure that reduction dimension tiling also works for smaller workloads.
    auto aType =
        gemmLikeOp.getDpsInputs()[0].getType().template cast<ShapedType>();
    auto kDim = aType.getShape().back();
    auto kTile = kDim < options.kTile ? kDim : options.kTile;

    auto dpasConfig = getDPASConfig(gemmLikeOp, kTile);
    if (!dpasConfig) {
      return rewriter.notifyMatchFailure(
          gemmLikeOp, "GEMM-like compute does not fit in DPAS tiles");
    }

    return gemmToDPAS(gemmLikeOp, *dpasConfig, kTile, rewriter);
  }

private:
  LinalgToXeGPUOptions options;
};

void populateLinalgToGpuPatterns(RewritePatternSet &patterns,
                                 LinalgToXeGPUOptions options) {
  patterns.add<ConvertGemmLikeToGpu<linalg::MatmulOp>,
               ConvertGemmLikeToGpu<linalg::BatchReduceMatmulOp>>(
      patterns.getContext(), options);
}

struct LinalgToXeGPU : public tpp::impl::LinalgToXeGPUBase<LinalgToXeGPU> {
  using LinalgToXeGPUBase::LinalgToXeGPUBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateLinalgToGpuPatterns(patterns, LinalgToXeGPUOptions{kTile});
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace
