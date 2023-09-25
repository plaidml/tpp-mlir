//===- LinalgToXegpu.cpp -----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"

#include "TPP/Dialect/XeGPU/IR/XeGPUOps.h"
#include "TPP/ValueUtils.h"

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

using namespace mlir;
using namespace mlir::tpp;

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
  assert(blockDims.size() <= 3 && "Too many GPU block dimensions");

  auto loc = op->getLoc();

  auto parentOp = op->getParentOp();
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

// Create XeGPU instructions out of matmul-like operation.
static LogicalResult gemmToXegpu(linalg::LinalgOp linalgOp,
                                 PatternRewriter &rewriter) {
  assert((isa_and_nonnull<linalg::MatmulOp>(linalgOp) ||
          isa_and_nonnull<linalg::BatchReduceMatmulOp>(linalgOp)) &&
         "Requires a matmul like op for XeGPU lowering");

  Location loc = linalgOp.getLoc();

  // If there is no parallel loop, create a unit blocks wrapper around the
  // current op. This allows for kernel outlining later on.
  auto blocksLoop = createGpuBlocksWrapper(linalgOp, {1, 1}, rewriter);
  if (blocksLoop)
    rewriter.setInsertionPoint(blocksLoop->getBody()->getTerminator());

  auto matA = linalgOp.getDpsInputOperands()[0]->get();
  auto matB = linalgOp.getDpsInputOperands()[1]->get();
  auto matC = linalgOp.getDpsInitOperands()[0]->get();

  auto typeA = matA.getType().cast<ShapedType>();
  auto typeB = matB.getType().cast<ShapedType>();
  auto typeC = matC.getType().cast<ShapedType>();

  // Skip batch dimension stride in case of brgemm.
  auto tileTypeA = xegpu::TileType::get(typeA.getShape().take_back(2),
                                        typeA.getElementType());
  auto tileTypeB = xegpu::TileType::get(typeB.getShape().take_back(2),
                                        typeB.getElementType());
  auto tileTypeC =
      xegpu::TileType::get(typeC.getShape(), typeC.getElementType());

  bool isBrgemm = isa<linalg::BatchReduceMatmulOp>(linalgOp);

  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

  OpBuilder::InsertionGuard guard(rewriter);

  // Fetch the inital value of the output element.
  auto tileC = rewriter.create<xegpu::InitTileOp>(
      loc, tileTypeC, matC, ValueRange{}, ValueRange{}, ValueRange{},
      SmallVector<int64_t>{0, 0}, tileTypeC.getShape(),
      SmallVector<int64_t>{1, 1});
  auto vecTypeC =
      VectorType::get(tileTypeC.getShape(), tileTypeC.getElementType());

  // No operands need transposition for now. Just present for API to be happy.
  auto transpose = BoolAttr::get(rewriter.getContext(), false);

  auto vnniAxisC = IntegerAttr::get(rewriter.getI32Type(), 0);
  Value loadC = rewriter.create<xegpu::Load2DOp>(loc, vecTypeC, tileC,
                                                 vnniAxisC, transpose);

  scf::ForOp batchLoop;
  Value batchIv;
  if (isBrgemm) {
    Value batch =
        rewriter.create<arith::ConstantIndexOp>(loc, typeA.getShape()[0]);
    batchLoop =
        rewriter.create<scf::ForOp>(loc, zero, batch, one, ValueRange{loadC});
    rewriter.setInsertionPointToStart(batchLoop.getBody());
    batchIv = batchLoop.getInductionVar();
    loadC = batchLoop.getRegionIterArg(0);
  }

  SmallVector<Value> inputStrides{one, one};
  if (isBrgemm)
    inputStrides.push_back(one);

  // In case of brgemm, the batch offset is dynamic.
  SmallVector<int64_t> staticOffsets;
  if (isBrgemm)
    staticOffsets.push_back(ShapedType::kDynamic);
  staticOffsets.push_back(0);
  staticOffsets.push_back(0);

  // Always assume unit strides for loading. If a buffer is strided, the info
  // will be encoded in the input memref type directly e.g., memref subview.
  SmallVector<int64_t> staticStrides{1, 1};
  if (isBrgemm)
    staticStrides.push_back(1);

  // Loaded tile is rank reduced in the batch dimension in case of brgemm.
  SmallVector<int64_t> staticSizesA;
  if (isBrgemm)
    staticSizesA.push_back(1);
  staticSizesA.push_back(tileTypeA.getShape()[0]);
  staticSizesA.push_back(tileTypeA.getShape()[1]);
  SmallVector<int64_t> staticSizesB;
  if (isBrgemm)
    staticSizesB.push_back(1);
  staticSizesB.push_back(tileTypeB.getShape()[0]);
  staticSizesB.push_back(tileTypeB.getShape()[1]);

  constexpr int vnniFactor = 2;

  auto tileA = rewriter.create<xegpu::InitTileOp>(
      loc, tileTypeA, matA, isBrgemm ? ValueRange{batchIv} : ValueRange{},
      ValueRange{}, ValueRange{}, staticOffsets, staticSizesA, staticStrides);
  auto shapeA = tileTypeA.getShape();
  auto vecTypeA =
      VectorType::get({shapeA[0], shapeA[1] / vnniFactor, vnniFactor},
                      tileTypeA.getElementType());

  auto tileB = rewriter.create<xegpu::InitTileOp>(
      loc, tileTypeB, matB, isBrgemm ? ValueRange{batchIv} : ValueRange{},
      ValueRange{}, ValueRange{}, staticOffsets, staticSizesB, staticStrides);
  auto shapeB = tileTypeB.getShape();
  auto vecTypeB =
      VectorType::get({shapeB[0] / vnniFactor, shapeB[1], vnniFactor},
                      tileTypeB.getElementType());

  auto vnniAxisA = IntegerAttr::get(rewriter.getI32Type(), 0);
  auto vnniAxisB = IntegerAttr::get(rewriter.getI32Type(), 1);

  auto loadA = rewriter.create<xegpu::Load2DOp>(loc, vecTypeA, tileA, vnniAxisA,
                                                transpose);
  auto loadB = rewriter.create<xegpu::Load2DOp>(loc, vecTypeB, tileB, vnniAxisB,
                                                transpose);

  Value result =
      rewriter.create<xegpu::DpasOp>(loc, vecTypeC, loadA, loadB, loadC);

  if (isBrgemm) {
    rewriter.setInsertionPointToEnd(batchLoop.getBody());
    rewriter.create<scf::YieldOp>(loc, ValueRange{result});
    result = batchLoop.getResults()[0];
    rewriter.setInsertionPointAfter(batchLoop);
  }

  // Write back the total sum to the output buffer.
  rewriter.create<xegpu::Store2DOp>(loc, tileC, result);

  rewriter.eraseOp(linalgOp);

  return success();
}

// Convert linalg.matmul to XeGPU kernel.
struct ConvertGemmToXegpu : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  ConvertGemmToXegpu(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    if (!matmulOp.hasBufferSemantics()) {
      return rewriter.notifyMatchFailure(
          matmulOp, "Linalg gemm to GPU expects memref type");
    }
    if (matmulOp.hasDynamicShape()) {
      return rewriter.notifyMatchFailure(
          matmulOp, "Expect static shape when mapping to GPU");
    }

    // TODO: add check if linalg op shapes and types are supported by XeGPU
    // (like in WMMA).
    return gemmToXegpu(matmulOp, rewriter);
  }
};

// Convert linalg.batch_reduce_matmul to XeGPU kernel.
struct ConvertBrgemmToXegpu
    : public OpRewritePattern<linalg::BatchReduceMatmulOp> {
  using OpRewritePattern<linalg::BatchReduceMatmulOp>::OpRewritePattern;

  ConvertBrgemmToXegpu(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(linalg::BatchReduceMatmulOp brgemmOp,
                                PatternRewriter &rewriter) const override {
    if (!brgemmOp.hasBufferSemantics()) {
      return rewriter.notifyMatchFailure(
          brgemmOp, "Linalg brgemm to GPU expects memref type");
    }
    if (brgemmOp.hasDynamicShape()) {
      return rewriter.notifyMatchFailure(
          brgemmOp, "Expect static shape when mapping to GPU");
    }

    // TODO: add check if linalg op shapes and types are supported by XeGPU
    // (like in WMMA).
    return gemmToXegpu(brgemmOp, rewriter);
  }
};

void populateLinalgToXegpuPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertGemmToXegpu, ConvertBrgemmToXegpu>(patterns.getContext());
}

struct LinalgToXegpu : public tpp::impl::LinalgToXegpuBase<LinalgToXegpu> {
  LinalgToXegpu() = default;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateLinalgToXegpuPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace
