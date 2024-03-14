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
#include "TPP/IR/StructuredOpMatcher.h"
#include "TPP/Transforms/Utils/ValueUtils.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/TransformOps/Utils.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/TypeSwitch.h"

#include <numeric>
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

// Return DPAS tile sizes if the gemm-like operation fits DPAS hardware.
static std::optional<SmallVector<int64_t>>
getDPASConfig(linalg::LinalgOp linalgOp, int kTile) {
  if (!(isa<linalg::MatmulOp>(linalgOp) ||
        isa<linalg::BatchReduceMatmulOp>(linalgOp) ||
        isa<linalg::GenericOp>(linalgOp))) {
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

// Verify if linalg operands fulfill XeGPU constraints.
LogicalResult isValidMemrefOperand(linalg::LinalgOp linalgOp, Value operand,
                                   PatternRewriter &rewriter,
                                   unsigned maxDims = 2) {
  auto type = dyn_cast<MemRefType>(operand.getType());
  if (!type) {
    return rewriter.notifyMatchFailure(
        linalgOp, "Expect memref operand for XeGPU lowering");
  }

  if (type.getShape().size() > maxDims) {
    return rewriter.notifyMatchFailure(
        linalgOp, "Too high dimensionality for XeGPU operations");
  }

  auto strides = utils::getStaticStrides(operand);

  if (failed(strides)) {
    return rewriter.notifyMatchFailure(
        linalgOp, "Expect static strides for XeGPU lowering");
  }
  if (strides->back() != 1) {
    return rewriter.notifyMatchFailure(linalgOp,
                                       "Expect unit stride in the innermost "
                                       "dimension for XeGPU operations");
  }

  return success();
}

// Match and, if possible, lower a generic operation to an XeGPU compatible op.
// Returns the result of the lowered op or nullopt, otherwise.
static std::optional<Value> lowerGenericOp(linalg::GenericOp genericOp,
                                           ArrayRef<Value> operands,
                                           VectorType resType,
                                           PatternRewriter &rewriter) {
  Location loc = genericOp.getLoc();

  // Expect operands to be already loaded vectors.
  for (auto operand : operands) {
    if (!isa<VectorType>(operand.getType()))
      return std::nullopt;
  }

  if (structured_match::utils::isTwoDReluOp(genericOp, /*operands=*/nullptr)) {
    assert(operands.size() == 1 &&
           "Invalid number of operands for generic 2D ReLU");

    auto eltType = resType.getElementType();
    Value zeroConst;

    if (isa<FloatType>(eltType)) {
      auto floatType = cast<FloatType>(eltType);
      zeroConst = rewriter.create<arith::ConstantFloatOp>(
          loc, APFloat::getZero(floatType.getFloatSemantics()), floatType);
    } else if (isa<IntegerType>(eltType)) {
      zeroConst = rewriter.create<arith::ConstantIntOp>(loc, 0, eltType);
    } else {
      // Unhandled type. Bail out.
      return std::nullopt;
    }

    auto zeroVec =
        rewriter.create<vector::BroadcastOp>(loc, resType, zeroConst);

    return rewriter
        .create<arith::MaximumFOp>(loc, resType, operands[0], zeroVec)
        .getResult();
  }

  if (structured_match::utils::isTwoDAddOp(genericOp, /*operands=*/nullptr)) {
    assert(operands.size() == 2 &&
           "Invalid number of operands for generic 2D add");
    return rewriter
        .create<arith::AddFOp>(loc, resType, operands[0], operands[1])
        .getResult();
  }

  return std::nullopt;
}

// Lower an elementwise operation to an XeGPU compatible op.
// Returns the result of the lowered op or nullopt, otherwise.
static std::optional<Value> lowerEltwiseOp(linalg::LinalgOp linalgOp,
                                           ArrayRef<Value> operands,
                                           VectorType resType,
                                           PatternRewriter &rewriter) {
  Location loc = linalgOp.getLoc();

  // Expect operands to be already loaded vectors.
  for (auto operand : operands) {
    if (!isa<VectorType>(operand.getType()))
      return std::nullopt;
  }

  auto eltType = resType.getElementType();

  return llvm::TypeSwitch<Operation *, std::optional<Value>>(linalgOp)
      .Case([&](linalg::AbsOp absOp) -> std::optional<Value> {
        assert(operands.size() == 1 && "Invalid number of operands for abs");
        if (isa<FloatType>(eltType)) {
          return rewriter.create<math::AbsFOp>(loc, resType, operands[0])
              .getResult();
        }
        if (isa<IntegerType>(eltType)) {
          return rewriter.create<math::AbsIOp>(loc, resType, operands[0])
              .getResult();
        }
        // Unhandled type. Bail out.
        return std::nullopt;
      })
      .Case([&](linalg::AddOp addOp) -> std::optional<Value> {
        assert(operands.size() == 2 && "Invalid number of operands for add");
        if (isa<FloatType>(eltType)) {
          return rewriter
              .create<arith::AddFOp>(loc, resType, operands[0], operands[1])
              .getResult();
        }
        if (isa<IntegerType>(eltType)) {
          return rewriter
              .create<arith::AddIOp>(loc, resType, operands[0], operands[1])
              .getResult();
        }
        // Unhandled type. Bail out.
        return std::nullopt;
      })
      .Case([&](linalg::CeilOp ceilOp) -> std::optional<Value> {
        assert(operands.size() == 1 && "Invalid number of operands for ceil");
        return rewriter.create<math::CeilOp>(loc, resType, operands[0])
            .getResult();
      })
      .Case([&](linalg::DivOp divOp) -> std::optional<Value> {
        assert(operands.size() == 2 && "Invalid number of operands for div");
        if (isa<FloatType>(eltType)) {
          return rewriter
              .create<arith::DivFOp>(loc, resType, operands[0], operands[1])
              .getResult();
        }
        if (isa<IntegerType>(eltType)) {
          return rewriter
              .create<arith::DivSIOp>(loc, resType, operands[0], operands[1])
              .getResult();
        }
        // Unhandled type. Bail out.
        return std::nullopt;
      })
      .Case([&](linalg::DivUnsignedOp divUnsignedOp) -> std::optional<Value> {
        assert(operands.size() == 2 &&
               "Invalid number of operands for unsigned div");
        if (isa<IntegerType>(eltType)) {
          return rewriter
              .create<arith::DivUIOp>(loc, resType, operands[0], operands[1])
              .getResult();
        }
        // Unhandled type. Bail out.
        return std::nullopt;
      })
      .Case([&](linalg::ExpOp expOp) -> std::optional<Value> {
        assert(operands.size() == 1 && "Invalid number of operands for exp");
        return rewriter.create<math::ExpOp>(loc, resType, operands[0])
            .getResult();
      })
      .Case([&](linalg::FloorOp floorOp) -> std::optional<Value> {
        assert(operands.size() == 1 && "Invalid number of operands for floor");
        return rewriter.create<math::FloorOp>(loc, resType, operands[0])
            .getResult();
      })
      .Case([&](linalg::MaxOp maxOp) -> std::optional<Value> {
        assert(operands.size() == 2 && "Invalid number of operands for max");
        if (isa<FloatType>(eltType)) {
          return rewriter
              .create<arith::MaximumFOp>(loc, resType, operands[0], operands[1])
              .getResult();
        }
        if (isa<IntegerType>(eltType)) {
          if (eltType.isUnsignedInteger()) {
            return rewriter
                .create<arith::MaxUIOp>(loc, resType, operands[0], operands[1])
                .getResult();
          } else {
            return rewriter
                .create<arith::MaxSIOp>(loc, resType, operands[0], operands[1])
                .getResult();
          }
        }
        // Unhandled type. Bail out.
        return std::nullopt;
      })
      .Case([&](linalg::MulOp mulOp) -> std::optional<Value> {
        assert(operands.size() == 2 && "Invalid number of operands for mul");
        if (isa<FloatType>(eltType)) {
          return rewriter
              .create<arith::MulFOp>(loc, resType, operands[0], operands[1])
              .getResult();
        }
        if (isa<IntegerType>(eltType)) {
          return rewriter
              .create<arith::MulIOp>(loc, resType, operands[0], operands[1])
              .getResult();
        }
        // Unhandled type. Bail out.
        return std::nullopt;
      })
      .Case([&](linalg::NegfOp negfOp) -> std::optional<Value> {
        assert(operands.size() == 1 && "Invalid number of operands for negf");
        return rewriter.create<arith::NegFOp>(loc, resType, operands[0])
            .getResult();
      })
      .Case([&](linalg::SubOp subOp) -> std::optional<Value> {
        assert(operands.size() == 2 && "Invalid number of operands for sub");
        if (isa<FloatType>(eltType)) {
          return rewriter
              .create<arith::SubFOp>(loc, resType, operands[0], operands[1])
              .getResult();
        }
        if (isa<IntegerType>(eltType)) {
          return rewriter
              .create<arith::SubIOp>(loc, resType, operands[0], operands[1])
              .getResult();
        }
        // Unhandled type. Bail out.
        return std::nullopt;
      })
      .Case([&](linalg::GenericOp genericOp) -> std::optional<Value> {
        return lowerGenericOp(genericOp, operands, resType, rewriter);
      })
      .Default(
          [&](Operation *op) -> std::optional<Value> { return std::nullopt; });
}

// Fuse an elementwise consumer operation.
// Returns updated store ops or nullopt if the fusion fails.
static std::optional<SmallVector<xegpu::StoreNDOp>>
eltwiseFusion(linalg::LinalgOp rootOp, linalg::LinalgOp consumerOp,
              SmallVector<xegpu::StoreNDOp> rootStoreOps,
              PatternRewriter &rewriter) {
  assert(rootStoreOps.size() > 0 && "Requires at least one store op");

  Location loc = rootOp.getLoc();
  auto ctx = rootOp.getContext();

  auto rootOutput = rootOp.getDpsInits()[0];

  // Gather additional operands of the fused consumer.
  // This excludes the root's output which values are already loaded into
  // registers and accessible through the store ops.
  SmallVector<Value> extraOperands;
  for (auto operand : consumerOp.getDpsInputOperands()) {
    if (operand->get() != rootOutput)
      extraOperands.push_back(operand->get());
  }

  // Insert fused eltwise ops before the stores and later replace the stores
  // with a new results.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(rootStoreOps[0]);

  // Collect new results after fusion.
  SmallVector<Value> fusedRes;
  auto readCacheHint =
      xegpu::CacheReadHintAttr::get(ctx, xegpu::CacheReadHint::CACHED);

  // For each store op, take a corresponding slice from the consumer operands
  // and load them into registers.
  for (auto storeOp : rootStoreOps) {
    auto storedVal = storeOp.getValue();
    auto storeDesc = storeOp.getTensorDesc();
    auto descOp = cast<xegpu::CreateNdDescOp>(storeDesc.getDefiningOp());

    // Create descriptors for the extra operands.
    SmallVector<Value> tensorDescs;
    for (auto operand : extraOperands) {
      auto tensorDesc = rewriter
                            .create<xegpu::CreateNdDescOp>(
                                loc, storeDesc.getType(), operand,
                                descOp.getOffsets(), descOp.getShape(),
                                descOp.getStrides(), descOp.getStaticOffsets(),
                                descOp.getBoundaryCheck(), descOp.getMode())
                            .getResult();
      tensorDescs.push_back(tensorDesc);
    }

    // Operands for the consumer op.
    // This always includes the previous result held by the store op.
    // Load values of the extra operands into registers.
    SmallVector<Value> operands{storedVal};
    for (auto tensorDesc : tensorDescs) {
      auto loadedVec = rewriter.create<xegpu::LoadNDOp>(
          loc, storedVal.getType(), tensorDesc, /*vnni_axis=*/nullptr,
          /*transpose=*/nullptr,
          /*l1_hint=*/readCacheHint,
          /*l2_hint=*/readCacheHint, /*l3_hint=*/readCacheHint,
          storeOp.getMode());
      operands.push_back(loadedVec);
    }

    // Lower to a vectorized eltwise op.
    auto newRes = lowerEltwiseOp(
        consumerOp, operands, cast<VectorType>(storedVal.getType()), rewriter);
    if (!newRes)
      return std::nullopt;

    fusedRes.push_back(*newRes);
  }

  // Fusion must have failed, if number of new results is different.
  // Bail out.
  if (fusedRes.size() != rootStoreOps.size())
    return std::nullopt;

  // Store the new result.
  auto writeCacheHint =
      xegpu::CacheWriteHintAttr::get(ctx, xegpu::CacheWriteHint::WRITE_BACK);
  SmallVector<xegpu::StoreNDOp> newStores;

  for (size_t i = 0; i < rootStoreOps.size(); i++) {
    auto storeDesc = rootStoreOps[i].getTensorDesc();

    auto newStore = rewriter.create<xegpu::StoreNDOp>(
        loc, storeDesc, fusedRes[i],
        /*l1_hint=*/writeCacheHint,
        /*l2_hint=*/writeCacheHint,
        /*l3_hint=*/writeCacheHint, rootStoreOps[i].getMode());
    newStores.push_back(newStore);
  }

  // Replace store ops and cleanup standalone consumer.
  for (size_t i = 0; i < rootStoreOps.size(); i++)
    rewriter.replaceOp(rootStoreOps[i], newStores[i]);

  rewriter.eraseOp(consumerOp);

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
static SmallVector<xegpu::StoreNDOp>
fuseEltwiseConsumers(linalg::LinalgOp rootOp,
                     SmallVector<xegpu::StoreNDOp> rootStoreOps,
                     PatternRewriter &rewriter) {
  auto consumers = getFusableConsumers(rootOp);

  for (auto consumer : consumers) {
    std::optional<SmallVector<xegpu::StoreNDOp>> updatedStoreOps = std::nullopt;

    updatedStoreOps = eltwiseFusion(rootOp, consumer, rootStoreOps, rewriter);

    // Failed to fuse operation. Bail out.
    if (!updatedStoreOps)
      break;

    rootStoreOps = *updatedStoreOps;
  }

  return rootStoreOps;
}

// Get static GPU block sizes represented by a surrounding operation
// like a kernel launch or parallel loop.
// Returns known block sizes if they are all static or failure, otherwise.
static FailureOr<SmallVector<int64_t>> getStaticBlockSizes(Operation *op) {
  if (!op)
    return failure();

  auto getConstVal = [&](Value val) -> std::optional<int64_t> {
    if (auto constOp = val.getDefiningOp<arith::ConstantIndexOp>()) {
      return constOp.value();
    }
    return std::nullopt;
  };

  if (auto launchOp = dyn_cast<gpu::LaunchOp>(op)) {
    auto sizeX = getConstVal(launchOp.getBlockSizeX());
    auto sizeY = getConstVal(launchOp.getBlockSizeY());
    auto sizeZ = getConstVal(launchOp.getBlockSizeZ());
    if (!sizeX || !sizeY || !sizeZ)
      return failure();

    return SmallVector<int64_t>{*sizeX, *sizeY, *sizeZ};
  }

  // TODO: Remove when the lowering only occurs within a gpu.launch op.
  //       Manually computing this is brittle and duplicated parallel
  //       loops to gpu conversion.
  if (auto blockLoop = dyn_cast<scf::ParallelOp>(op)) {
    auto gridLoop = blockLoop->getParentOfType<scf::ParallelOp>();

    // Blocks or number of threads are represented by the first parallel loop
    // nested within another parallel loop.
    //
    // Fail if there is no outer parallel loop or current loop is nested more
    // than once.
    if (!gridLoop || (gridLoop->getParentOfType<scf::ParallelOp>())) {
      return failure();
    }

    SmallVector<int64_t> blockSizes;
    for (auto [lb, ub, step] :
         llvm::zip_equal(blockLoop.getLowerBound(), blockLoop.getUpperBound(),
                         blockLoop.getStep())) {
      auto lbVal = getConstVal(lb);
      auto ubVal = getConstVal(ub);
      auto stepVal = getConstVal(step);
      if (!lbVal || !ubVal || !stepVal)
        return failure();

      int64_t blockSize = (*ubVal - *lbVal) / *stepVal;

      // Assume that at least one thread/workitem is created for the given
      // dimension. Otherwise, outlining will fail anyway.
      blockSizes.push_back(blockSize < 0 ? 1 : blockSize);
    }

    // Too many dimensions, something went wrong. Bail out.
    if (blockSizes.size() > 3)
      return failure();

    return blockSizes;
  }

  return failure();
}

static Value getGpuLinearThreadId(PatternRewriter &rewriter, Location loc) {
  SmallVector<Value, 3> threadIds;
  SmallVector<Value, 3> blockDims;

  for (auto dim : {gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z}) {
    threadIds.push_back(rewriter.create<gpu::ThreadIdOp>(loc, dim));
    blockDims.push_back(rewriter.create<gpu::BlockDimOp>(loc, dim));
  }

  // The default GPU indexing is modeled after CUDA:
  // linear index = (z * sizeY + y) * sizeX + x
  Value threadId =
      rewriter.create<arith::MulIOp>(loc, threadIds[2], blockDims[1]);
  threadId = rewriter.create<arith::AddIOp>(loc, threadId, threadIds[1]);
  threadId = rewriter.create<arith::MulIOp>(loc, threadId, blockDims[0]);
  threadId = rewriter.create<arith::AddIOp>(loc, threadId, threadIds[0]);

  return threadId;
}

static xegpu::CreateNdDescOp
createGemmPrefetchTile(PatternRewriter &rewriter, linalg::LinalgOp linalgOp,
                       unsigned inputPos, int64_t numThreads,
                       ArrayRef<int> blockTile, ArrayRef<int> threadTile,
                       xegpu::Mode mode, Value batchIv) {
  assert(inputPos <= 1 && "Can handle only GEMM inputs: mat A or mat B");
  Location loc = linalgOp.getLoc();

  Value src = linalgOp.getDpsInputs()[inputPos];

  const int tileRows = blockTile[0];
  const int tileCols = blockTile[1];

  const int numElements = tileRows * tileCols;
  const int elementsPerThread = numElements / numThreads;

  // Prioritize loading the whole block tile dimension.
  // The shared reduction dimension will be spread across the workers.
  int numRows = threadTile[0];
  int numCols = elementsPerThread / numRows;
  if (inputPos == 1) {
    numCols = threadTile[1];
    numRows = elementsPerThread / numCols;
  }

  auto srcType = src.getType().cast<ShapedType>();

  auto prefetchType =
      xegpu::TensorDescType::get({numRows, numCols}, srcType.getElementType());

  Value threadId = getGpuLinearThreadId(rewriter, loc);

  Value numColTiles =
      rewriter.create<arith::ConstantIndexOp>(loc, tileCols / numCols);
  Value tileRowOffset =
      rewriter.create<arith::DivUIOp>(loc, threadId, numColTiles);
  Value tileColOffset =
      rewriter.create<arith::RemUIOp>(loc, threadId, numColTiles);

  Value tileRowSize = rewriter.create<arith::ConstantIndexOp>(loc, numRows);
  Value tileColSize = rewriter.create<arith::ConstantIndexOp>(loc, numCols);
  Value eltRowOffset =
      rewriter.create<arith::MulIOp>(loc, tileRowOffset, tileRowSize);
  Value eltColOffset =
      rewriter.create<arith::MulIOp>(loc, tileColOffset, tileColSize);

  SmallVector<mlir::OpFoldResult> prefetchOffsets;
  if (batchIv)
    prefetchOffsets.push_back(batchIv);
  prefetchOffsets.append({eltRowOffset, eltColOffset});

  return rewriter.create<xegpu::CreateNdDescOp>(loc, prefetchType, src,
                                                prefetchOffsets,
                                                /*boundary_check=*/true, mode);
}

static SmallVector<Value>
prefetchAndUpdateTiles(PatternRewriter &rewriter, Location loc, Value prefetchA,
                       Value prefetchB, Value kTileOffset,
                       xegpu::CacheReadHintAttr readCacheHint,
                       xegpu::Mode mode) {
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

  // Prefetch the next set of input tiles.
  rewriter.create<xegpu::PrefetchNDOp>(loc, prefetchA,
                                       /*l1_hint=*/readCacheHint,
                                       /*l2_hint=*/readCacheHint,
                                       /*l3_hint=*/readCacheHint, mode);
  rewriter.create<xegpu::PrefetchNDOp>(loc, prefetchB,
                                       /*l1_hint=*/readCacheHint,
                                       /*l2_hint=*/readCacheHint,
                                       /*l3_hint=*/readCacheHint, mode);

  // Update offsets of the prefetch tiles.
  // Shift along the reduction dimension.
  auto updatedTileA =
      rewriter
          .create<xegpu::UpdateNDOffsetOp>(loc, prefetchA.getType(), prefetchA,
                                           ValueRange{zero, kTileOffset}, mode)
          .getResult();
  auto updateTileB =
      rewriter
          .create<xegpu::UpdateNDOffsetOp>(loc, prefetchB.getType(), prefetchB,
                                           ValueRange{kTileOffset, zero}, mode)
          .getResult();

  return SmallVector<Value>{updatedTileA, updateTileB};
}

// Create XeGPU DPAS kernel out of GEMM-like operation.
static LogicalResult createDPASKernel(linalg::LinalgOp linalgOp,
                                      ArrayRef<int64_t> dpasTile, int kTile,
                                      PatternRewriter &rewriter) {
  assert((isa<linalg::MatmulOp>(linalgOp) ||
          isa<linalg::BatchReduceMatmulOp>(linalgOp) ||
          isa<linalg::GenericOp>(linalgOp)) &&
         "Requires a GEMM-like op for DPAS lowering");

  Location loc = linalgOp.getLoc();
  auto ctx = linalgOp.getContext();

  auto matA = linalgOp.getDpsInputs()[0];
  auto matB = linalgOp.getDpsInputs()[1];
  auto matC = linalgOp.getDpsInits()[0];

  auto typeA = matA.getType().cast<ShapedType>();
  auto typeB = matB.getType().cast<ShapedType>();
  auto typeC = matC.getType().cast<ShapedType>();

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

  // Create prefetch tiles.
  //
  // Fewer prefetch requests decrese memory controller pressure and allow
  // more efficient data loading.
  //
  // TODO: Add support for multistage prefetching.
  int64_t numThreads = 1;
  auto blockDims =
      getStaticBlockSizes(linalgOp->getParentOfType<scf::ParallelOp>());
  if (succeeded(blockDims)) {
    numThreads = std::accumulate(blockDims->begin(), blockDims->end(), 1,
                                 std::multiplies<int64_t>());
  }
  // Disable prefetching when there is no block/workgroup parallelism.
  bool isCoopPrefetch = numThreads > 1;

  Value prefetchA;
  Value prefetchB;
  xegpu::TensorDescType prefetchTypeA;
  xegpu::TensorDescType prefetchTypeB;
  Value kTileOffset = rewriter.create<arith::ConstantIndexOp>(loc, kTile);
  if (isCoopPrefetch) {
    // Return dimension size on which the whole block/workgroup operates.
    auto getBlockLevelSize = [&](Value val, int dim) -> int {
      if (auto subview =
              dyn_cast_or_null<memref::SubViewOp>(val.getDefiningOp())) {
        val = subview.getSource();
      }

      return cast<ShapedType>(val.getType()).getShape()[dim];
    };

    int blockRows = getBlockLevelSize(matC, 0);
    int blockCols = getBlockLevelSize(matC, 1);

    auto prefetchDescA = createGemmPrefetchTile(
        rewriter, linalgOp, /*inputPos=*/0, numThreads, {blockRows, kTile},
        {dimM, kTile}, xegpuMode, batchIv);
    prefetchA = prefetchDescA.getResult();
    prefetchTypeA = prefetchDescA.getType();

    auto prefetchDescB = createGemmPrefetchTile(
        rewriter, linalgOp, /*inputPos=*/1, numThreads, {kTile, blockCols},
        {kTile, dimN}, xegpuMode, batchIv);
    prefetchB = prefetchDescB.getResult();
    prefetchTypeB = prefetchDescB.getType();

    auto updatedTiles =
        prefetchAndUpdateTiles(rewriter, loc, prefetchA, prefetchB, kTileOffset,
                               readCacheHint, xegpuMode);
    prefetchA = updatedTiles[0];
    prefetchB = updatedTiles[1];

    // Ensure that block/workgroup is sychronized after prefetching.
    rewriter.create<xegpu::CompileHintOp>(loc);
    rewriter.create<gpu::BarrierOp>(loc);
  }

  // Construct and move into GEMM reduction dimension tiling loop.
  // Propagate output values as iter args.
  SmallVector<Value> iterArgs;
  iterArgs.append(loadVecC);
  iterArgs.append(tilesA);
  iterArgs.append(tilesB);
  if (isCoopPrefetch) {
    iterArgs.push_back(prefetchA);
    iterArgs.push_back(prefetchB);
  }
  scf::ForOp kDimLoop = startLoop(0, dimK, kTile, iterArgs);
  auto iterValues = getLoopIterValues(kDimLoop);

  loadVecC = SmallVector<Value>{iterValues.begin(),
                                iterValues.begin() + loadVecC.size()};
  tilesA =
      SmallVector<Value>{iterValues.begin() + loadVecC.size(),
                         iterValues.begin() + loadVecC.size() + tilesA.size()};
  tilesB = SmallVector<Value>{iterValues.begin() + loadVecC.size() + tilesA.size(),
                              iterValues.begin() + loadVecC.size() + tilesA.size() + tilesB.size()};
  if (isCoopPrefetch) {
    prefetchA = *(iterValues.end() - 2);
    prefetchB = *(iterValues.end() - 1);
  }

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

  if (isCoopPrefetch) {
    auto updatedTiles =
        prefetchAndUpdateTiles(rewriter, loc, prefetchA, prefetchB, kTileOffset,
                               readCacheHint, xegpuMode);
    prefetchA = updatedTiles[0];
    prefetchB = updatedTiles[1];
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
  if (isCoopPrefetch) {
    yieldVals.push_back(prefetchA);
    yieldVals.push_back(prefetchB);
  }

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

  (void)fuseEltwiseConsumers(linalgOp, storeOps, rewriter);

  rewriter.eraseOp(linalgOp);

  return success();
}

// Create XeGPU kernel out of elementwise operation.
LogicalResult createEltwiseKernel(linalg::LinalgOp linalgOp,
                                  ArrayRef<int64_t> tileSizes,
                                  PatternRewriter &rewriter) {
  assert(tileSizes.size() == 2 && "Require 2D tile size for eltwise lowering");

  Location loc = linalgOp.getLoc();
  auto ctx = linalgOp.getContext();

  auto output = linalgOp.getDpsInits()[0];
  auto outputType = output.getType().cast<ShapedType>();
  auto outputShape = outputType.getShape();

  bool isOutput2D = outputShape.size() == 2;

  auto dimM = outputShape[0];
  auto dimN = isOutput2D ? outputShape[1] : 0;

  SmallVector<int64_t> eltwiseTileShape{tileSizes[0]};
  if (isOutput2D)
    eltwiseTileShape.push_back(tileSizes[1]);

  // Linalg named elementwise operations guarantee that all operands
  // have the same shape and type. Thus, the same tensor descriptor
  // type can be used for all operands.
  auto tensorDesc = xegpu::TensorDescType::get(
      eltwiseTileShape, output.getType().cast<ShapedType>().getElementType());
  auto xegpuMode = xegpu::Mode::VC;

  // Create tiled tensor descriptors.
  auto createDescs = [&](Value source) -> SmallVector<Value> {
    SmallVector<Value> tiles;
    // TODO: Use larger tile size in case of 1D inputs.
    for (int m = 0; m < dimM; m += tileSizes[0]) {
      Value rowIdx = rewriter.create<arith::ConstantIndexOp>(loc, m);

      // Flexibly create 1D or 2D descriptors.
      int n = 0;
      do {
        mlir::SmallVector<mlir::OpFoldResult> loadOffsets{rowIdx};

        if (dimN > 0) {
          Value colIdx = rewriter.create<arith::ConstantIndexOp>(loc, n);
          loadOffsets.push_back(colIdx);
        }

        Value tile = rewriter
                         .create<xegpu::CreateNdDescOp>(
                             loc, tensorDesc, source, loadOffsets,
                             /*boundary_check=*/true, xegpuMode)
                         .getResult();
        tiles.push_back(tile);

        n += tileSizes[1];
      } while (n < dimN);
    }

    return tiles;
  };

  // Output descriptors for later stores.
  SmallVector<Value> outputTiles = createDescs(output);

  // Create descriptors and load values for all inputs.
  auto vecType =
      VectorType::get(tensorDesc.getShape(), tensorDesc.getElementType());

  SmallVector<SmallVector<Value>> loadedInputs;
  for (auto input : linalgOp.getDpsInputs()) {
    SmallVector<Value> inputTiles = createDescs(input);
    SmallVector<Value> loadedVals;
    for (auto tile : inputTiles) {
      auto loadedVec = rewriter.create<xegpu::LoadNDOp>(
          loc, vecType, tile, /*vnni_axis=*/nullptr,
          /*transpose=*/nullptr,
          /*l1_hint=*/nullptr,
          /*l2_hint=*/nullptr, /*l3_hint=*/nullptr, xegpuMode);
      loadedVals.push_back(loadedVec);
    }
    loadedInputs.push_back(loadedVals);
  }

  // Perform vectorized computations for each output tile.
  SmallVector<Value> results;
  for (size_t i = 0; i < outputTiles.size(); i++) {
    SmallVector<Value> operands;
    for (auto inputs : loadedInputs) {
      operands.push_back(inputs[i]);
    }
    auto res = lowerEltwiseOp(linalgOp, operands, vecType, rewriter);
    if (!res)
      return failure();

    results.push_back(*res);
  }

  // Store results.
  auto writeCacheHint =
      xegpu::CacheWriteHintAttr::get(ctx, xegpu::CacheWriteHint::WRITE_BACK);
  for (size_t i = 0; i < outputTiles.size(); i++) {
    rewriter.create<xegpu::StoreNDOp>(loc, outputTiles[i], results[i],
                                      /*l1_hint=*/writeCacheHint,
                                      /*l2_hint=*/writeCacheHint,
                                      /*l3_hint=*/writeCacheHint, xegpuMode);
  }

  rewriter.eraseOp(linalgOp);

  return success();
}

// Convert a GEMM-like operation to an XeGPU kernel.
template <typename LinalgOpTy>
struct ConvertGemmLikeToXeGPU : public OpRewritePattern<LinalgOpTy> {
  using OpRewritePattern<LinalgOpTy>::OpRewritePattern;
  // Constrain conversion to the supported GEMM-like ops.
  static_assert(
      llvm::is_one_of<LinalgOpTy, linalg::MatmulOp, linalg::BatchReduceMatmulOp,
                      linalg::GenericOp>::value);

  ConvertGemmLikeToXeGPU(MLIRContext *ctx, LinalgToXeGPUOptions options)
      : OpRewritePattern<LinalgOpTy>(ctx), options(options) {}

  LogicalResult matchAndRewrite(LinalgOpTy gemmLikeOp,
                                PatternRewriter &rewriter) const override {
    if (!gemmLikeOp.hasPureBufferSemantics()) {
      return rewriter.notifyMatchFailure(
          gemmLikeOp, "Linalg GEMM-like to GPU expects memref type");
    }
    if (gemmLikeOp.hasDynamicShape()) {
      return rewriter.notifyMatchFailure(
          gemmLikeOp, "Expect static shape when mapping to GPU");
    }

    using namespace structured_match;
    auto matmulMatcher =
        StructuredOpMatcher::make<linalg::GenericOp>()
            .operation(NumDpsInits(EqualsTo(1)))
            .operation(NumDpsInputs(EqualsTo(2)))
            .operation(NumRegions(EqualsTo(1)))
            .operation(NumOfLoops(EqualsTo(3)))
            .input(MatchAll(), HasStaticShape())
            .output(MatchAll(), HasStaticShape())
            .region(MatchOne(0), WithOpChain<arith::MulFOp, arith::AddFOp>());
    if (isa<linalg::GenericOp>(gemmLikeOp) &&
        !matmulMatcher.match(gemmLikeOp)) {
      return rewriter.notifyMatchFailure(
          gemmLikeOp, "Generic does not represent a GEMM-like operation");
    }

    for (auto input : gemmLikeOp.getDpsInputs()) {
      // 3D inputs are also acceptable in case of brgemm.
      auto isInputValid =
          isValidMemrefOperand(gemmLikeOp, input, rewriter, /*maxDims=*/3);
      if (failed(isInputValid))
        return isInputValid;
    }
    auto isOutputValid =
        isValidMemrefOperand(gemmLikeOp, gemmLikeOp.getDpsInits()[0], rewriter);
    if (failed(isOutputValid))
      return isOutputValid;

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

    return createDPASKernel(gemmLikeOp, *dpasConfig, kTile, rewriter);
  }

private:
  LinalgToXeGPUOptions options;
};

// Convert a named elementwise operation to an XeGPU kernel.
template <typename LinalgOpTy>
struct ConvertNamedEltwiseToXeGPU : public OpRewritePattern<LinalgOpTy> {
  using OpRewritePattern<LinalgOpTy>::OpRewritePattern;

  ConvertNamedEltwiseToXeGPU(MLIRContext *ctx, LinalgToXeGPUOptions options)
      : OpRewritePattern<LinalgOpTy>(ctx), options(options) {}

  LogicalResult matchAndRewrite(LinalgOpTy eltwiseOp,
                                PatternRewriter &rewriter) const override {
    if (!eltwiseOp.hasPureBufferSemantics()) {
      return rewriter.notifyMatchFailure(
          eltwiseOp, "Linalg eltwise to GPU expects memref type");
    }
    if (eltwiseOp.hasDynamicShape()) {
      return rewriter.notifyMatchFailure(
          eltwiseOp, "Expect static shape when mapping to GPU");
    }

    for (auto input : eltwiseOp.getDpsInputs()) {
      auto isInputValid = isValidMemrefOperand(eltwiseOp, input, rewriter);
      if (failed(isInputValid))
        return isInputValid;
    }
    auto isOutputValid =
        isValidMemrefOperand(eltwiseOp, eltwiseOp.getDpsInits()[0], rewriter);
    if (failed(isOutputValid))
      return isOutputValid;

    // TODO: Tile sizes for vectorized eltwise operations should be chosen
    //       dynamically based on the workload and target hardware.
    SmallVector<int64_t> tileSizes{8, 16};

    return createEltwiseKernel(eltwiseOp, tileSizes, rewriter);
  }

private:
  LinalgToXeGPUOptions options;
};

void populateLinalgGemmToXeGPUPatterns(RewritePatternSet &patterns,
                                       LinalgToXeGPUOptions options) {
  patterns.add<ConvertGemmLikeToXeGPU<linalg::MatmulOp>,
               ConvertGemmLikeToXeGPU<linalg::BatchReduceMatmulOp>,
               ConvertGemmLikeToXeGPU<linalg::GenericOp>>(patterns.getContext(),
                                                          options);
}

void populateLinalgEltwiseToXeGPUPatterns(RewritePatternSet &patterns,
                                          LinalgToXeGPUOptions options) {
  patterns.add<ConvertNamedEltwiseToXeGPU<linalg::AbsOp>,
               ConvertNamedEltwiseToXeGPU<linalg::AddOp>,
               ConvertNamedEltwiseToXeGPU<linalg::CeilOp>,
               ConvertNamedEltwiseToXeGPU<linalg::DivOp>,
               ConvertNamedEltwiseToXeGPU<linalg::DivUnsignedOp>,
               ConvertNamedEltwiseToXeGPU<linalg::ExpOp>,
               ConvertNamedEltwiseToXeGPU<linalg::FloorOp>,
               ConvertNamedEltwiseToXeGPU<linalg::MaxOp>,
               ConvertNamedEltwiseToXeGPU<linalg::MulOp>,
               ConvertNamedEltwiseToXeGPU<linalg::NegfOp>,
               ConvertNamedEltwiseToXeGPU<linalg::SubOp>>(patterns.getContext(),
                                                          options);
}

struct LinalgToXeGPU : public tpp::impl::LinalgToXeGPUBase<LinalgToXeGPU> {
  using LinalgToXeGPUBase::LinalgToXeGPUBase;

  void runOnOperation() override {
    LinalgToXeGPUOptions options{kTile};

    // Run GEMM pattern first to allow fusion with its consumers.
    RewritePatternSet gemmPatterns(&getContext());
    populateLinalgGemmToXeGPUPatterns(gemmPatterns, options);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(gemmPatterns));

    // Convert other remaining ops.
    RewritePatternSet patterns(&getContext());
    populateLinalgEltwiseToXeGPUPatterns(patterns, options);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace
