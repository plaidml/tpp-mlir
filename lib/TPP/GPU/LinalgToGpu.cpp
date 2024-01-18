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

static int getComputeCapability(llvm::StringRef chip) {
  llvm::StringRef delim = "_";
  llvm::StringRef ccToken =
      chip.substr(chip.find(delim) + delim.size(), chip.size());

  return std::stoi(ccToken.str());
}

// Return true if hardware supports WMMA operations.
static bool isMMASupported(llvm::StringRef triple, llvm::StringRef chip) {
  return (triple == "nvptx64-nvidia-cuda") &&
         (getComputeCapability(chip) >= 70);
}

struct WMMASettings {
  int m;
  int n;
  int k;
};

// Return true if the operation can be represented with WMMA compute.
static std::optional<WMMASettings> getWMMASettings(linalg::LinalgOp linalgOp) {
  if (!(isa_and_nonnull<linalg::MatmulOp>(linalgOp) ||
        isa_and_nonnull<linalg::BatchReduceMatmulOp>(linalgOp))) {
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

  // TODO: Add more possible tile sizes and choose optimal one.
  int wmmaTileM = 16;
  int wmmaTileN = 16;
  int wmmaTileK = 16;
  if ((mDim % wmmaTileM == 0) && (nDim % wmmaTileN == 0) &&
      (kDim % wmmaTileK == 0))
    return WMMASettings{wmmaTileM, wmmaTileN, wmmaTileK};

  return std::nullopt;
}

// Fuse a consumer using WMMA operations.
// Returns updated store op or nullopt if the fusion fails.
static std::optional<Operation *>
mmaFusion(linalg::LinalgOp rootOp, linalg::LinalgOp consumer,
          gpu::SubgroupMmaStoreMatrixOp rootStoreOp, ValueRange storeIndices,
          PatternRewriter &rewriter) {
  Location loc = rootOp.getLoc();

  auto rootOutput = rootOp.getDpsInits()[0];
  auto outputType = rootOutput.getType().cast<ShapedType>();
  // Must be a floating point type.
  auto floatType = dyn_cast<FloatType>(outputType.getElementType());
  if (!floatType)
    return std::nullopt;

  gpu::MMAMatrixType mmaOutputType = gpu::MMAMatrixType::get(
      outputType.getShape(), outputType.getElementType(), "COp");
  auto leadingDim = rootStoreOp.getLeadDimension();

  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

  // Insert fused eltwise ops before the store and later replace the store
  // with a new result.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(rootStoreOp);

  std::optional<Operation *> newStore = std::nullopt;
  SmallVector<Value> operands;
  if (structured_match::utils::isTwoDAddOp(consumer, &operands)) {
    // Get the value to be added - load the tile first.
    // Must be a buffer of the same type - scalar broadcast is not supported.
    auto addValue = (operands[0] != rootOutput) ? operands[0] : operands[1];
    if (addValue.getType() != rootOutput.getType())
      return std::nullopt;
    // Fuse the add into the matmul body.
    addValue = rewriter
                   .create<gpu::SubgroupMmaLoadMatrixOp>(
                       loc, mmaOutputType, addValue, ValueRange{zero, zero},
                       leadingDim,
                       /*transpose=*/UnitAttr())
                   .getRes();
    auto eltwiseAttr = gpu::MMAElementwiseOp::ADDF;
    auto addRes =
        rewriter
            .create<gpu::SubgroupMmaElementwiseOp>(
                loc, mmaOutputType, ValueRange{rootStoreOp.getSrc(), addValue},
                eltwiseAttr)
            .getRes();
    // Store the new result.
    newStore = rewriter.replaceOpWithNewOp<gpu::SubgroupMmaStoreMatrixOp>(
        rootStoreOp, addRes, rootStoreOp.getDstMemref(), ValueRange{zero, zero},
        leadingDim,
        /*transpose=*/UnitAttr());
  } else if (structured_match::utils::isTwoDReluOp(consumer, &operands)) {
    // Fuse the relu into the matmul body.
    Value zeroFloat = rewriter.create<arith::ConstantFloatOp>(
        loc, APFloat::getZero(floatType.getFloatSemantics()), floatType);
    Value zeroTile = rewriter.create<gpu::SubgroupMmaConstantMatrixOp>(
        loc, mmaOutputType, zeroFloat);
    auto eltwiseAttr = gpu::MMAElementwiseOp::MAXF;
    auto maxRes =
        rewriter
            .create<gpu::SubgroupMmaElementwiseOp>(
                loc, mmaOutputType, ValueRange{rootStoreOp.getSrc(), zeroTile},
                eltwiseAttr)
            .getRes();
    // Store the new result.
    newStore = rewriter.replaceOpWithNewOp<gpu::SubgroupMmaStoreMatrixOp>(
        rootStoreOp, maxRes, rootStoreOp.getDstMemref(), ValueRange{zero, zero},
        leadingDim,
        /*transpose=*/UnitAttr());
  } else {
    // Not a fusable operation. Bail out.
    return std::nullopt;
  }

  rewriter.eraseOp(consumer);

  return newStore;
}

// Fuse a consumer using scalar operations.
// Returns updated store op or nullopt if the fusion fails.
static std::optional<Operation *> scalarFusion(linalg::LinalgOp rootOp,
                                               linalg::LinalgOp consumer,
                                               memref::StoreOp rootStoreOp,
                                               ValueRange storeIndices,
                                               PatternRewriter &rewriter) {
  Location loc = rootOp.getLoc();
  auto rootOutput = rootOp.getDpsInits()[0];
  auto outputType = rootOutput.getType().cast<ShapedType>();
  // Must be a floating point type.
  auto floatType = dyn_cast<FloatType>(outputType.getElementType());
  if (!floatType)
    return std::nullopt;

  // Insert fused eltwise ops before the store and later replace the store
  // with a new result.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(rootStoreOp);

  std::optional<Operation *> newStore = std::nullopt;
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

// Fuse elementwise consumers.
// A naive fusion strategy that looks at the other operations after the root
// linalg op and tries to fuse them.
// Attemps bails on the first mismatch.
// Returns updated store op.
static Operation *fuseEltwiseConsumers(linalg::LinalgOp rootOp,
                                       Operation *rootStoreOp,
                                       ValueRange storeIndices,
                                       PatternRewriter &rewriter) {
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
    if (outBuf != rootOutput)
      break;

    consumers.push_back(consumer);
  }

  for (auto op : consumers) {
    std::optional<Operation *> updatedStoreOp = std::nullopt;
    if (auto storeOp = dyn_cast<memref::StoreOp>(rootStoreOp)) {
      updatedStoreOp =
          scalarFusion(rootOp, op, storeOp, storeIndices, rewriter);
    } else if (auto mmaStore =
                   dyn_cast<gpu::SubgroupMmaStoreMatrixOp>(rootStoreOp)) {
      updatedStoreOp = mmaFusion(rootOp, op, mmaStore, storeIndices, rewriter);
    }

    // Not a fusable operation. Bail out.
    if (!updatedStoreOp)
      break;

    rootStoreOp = *updatedStoreOp;
  }

  return rootStoreOp;
}

// Create WMMA instructions out of matmul-like operation.
static LogicalResult gemmToGpuMMA(linalg::LinalgOp linalgOp,
                                  WMMASettings wmmaSettings,
                                  PatternRewriter &rewriter) {
  assert((isa_and_nonnull<linalg::MatmulOp>(linalgOp) ||
          isa_and_nonnull<linalg::BatchReduceMatmulOp>(linalgOp)) &&
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
  for (int i = 0; i < dimM; i += wmmaSettings.m) {
    for (int j = 0; j < dimN; j += wmmaSettings.n) {
      Value rowIdx = rewriter.create<arith::ConstantIndexOp>(loc, i);
      Value colIdx = rewriter.create<arith::ConstantIndexOp>(loc, j);
      Value tileC =
          rewriter
              .create<gpu::SubgroupMmaLoadMatrixOp>(
                  loc, mmaTypeC, matC, ValueRange{rowIdx, colIdx}, ldc,
                  /*transpose=*/UnitAttr())
              .getRes();
      tilesC.push_back(tileC);
    }
  }

  scf::ForOp batchLoop;
  Value batchIv;
  if (isBrgemm) {
    Value batch =
        rewriter.create<arith::ConstantIndexOp>(loc, typeA.getShape()[0]);
    batchLoop = rewriter.create<scf::ForOp>(loc, zero, batch, one, tilesC);
    rewriter.setInsertionPointToStart(batchLoop.getBody());
    batchIv = batchLoop.getInductionVar();

    SmallVector<Value> newTiles;
    for (auto iterArgTile : batchLoop.getRegionIterArgs())
      newTiles.push_back(iterArgTile);

    tilesC = newTiles;
  }

  SmallVector<Value> tilesA;
  for (int i = 0; i < dimM; i += wmmaSettings.m) {
    for (int j = 0; j < dimK; j += wmmaSettings.k) {
      Value rowIdx = rewriter.create<arith::ConstantIndexOp>(loc, i);
      Value colIdx = rewriter.create<arith::ConstantIndexOp>(loc, j);
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

  SmallVector<Value> tilesB;
  for (int i = 0; i < dimK; i += wmmaSettings.k) {
    for (int j = 0; j < dimN; j += wmmaSettings.n) {
      Value rowIdx = rewriter.create<arith::ConstantIndexOp>(loc, i);
      Value colIdx = rewriter.create<arith::ConstantIndexOp>(loc, j);
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

  SmallVector<Value> results;
  for (size_t i = 0; i < tilesC.size(); i++) {
    Value result =
        rewriter
            .create<gpu::SubgroupMmaComputeOp>(loc, tilesC[i].getType(),
                                               tilesA[i], tilesB[i], tilesC[i],
                                               /*a_transpose=*/UnitAttr(),
                                               /*b_transpose=*/UnitAttr())
            .getRes();
    results.push_back(result);
  }

  if (isBrgemm) {
    rewriter.setInsertionPointToEnd(batchLoop.getBody());
    rewriter.create<scf::YieldOp>(loc, results);
    results = batchLoop.getResults();
    rewriter.setInsertionPointAfter(batchLoop);
  }

  // Write back the total sum to the output buffer.
  SmallVector<Operation *> storeOps;
  for (int i = 0; i < dimM / wmmaSettings.m; i++) {
    for (int j = 0; j < dimN / wmmaSettings.n; j++) {
      int resIdx = (dimM / wmmaSettings.m) * i + j;

      Value rowIdx =
          rewriter.create<arith::ConstantIndexOp>(loc, i * wmmaSettings.m);
      Value colIdx =
          rewriter.create<arith::ConstantIndexOp>(loc, j * wmmaSettings.n);
      auto storeOp = rewriter.create<gpu::SubgroupMmaStoreMatrixOp>(
          loc, results[resIdx], matC, ValueRange{rowIdx, colIdx}, ldc,
          /*transpose=*/UnitAttr());
      storeOps.push_back(storeOp);
    }
  }

  (void)fuseEltwiseConsumers(linalgOp, storeOps[0], ValueRange{zero, zero},
                             rewriter);

  rewriter.eraseOp(linalgOp);

  return success();
}

// Create loops out of matmul-like operation.
static LogicalResult gemmToGpuLoops(linalg::LinalgOp linalgOp,
                                    PatternRewriter &rewriter) {
  assert((isa_and_nonnull<linalg::MatmulOp>(linalgOp) ||
          isa_and_nonnull<linalg::BatchReduceMatmulOp>(linalgOp)) &&
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

  (void)fuseEltwiseConsumers(linalgOp, storeOp, parallelIvs, rewriter);

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

    if (options.useWmma && isMMASupported(options.gpuTriple, options.gpuChip)) {
      if (auto settings = getWMMASettings(gemmLikeOp))
        return gemmToGpuMMA(gemmLikeOp, *settings, rewriter);
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
