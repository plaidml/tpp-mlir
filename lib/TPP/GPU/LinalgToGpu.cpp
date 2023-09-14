//===- LinalgToGpu.cpp -------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"

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

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

// Return true if the operation can be represented with WMMA compute.
bool supportsMMACompute(linalg::LinalgOp linalgOp) {
  if (!(isa_and_nonnull<linalg::MatmulOp>(linalgOp) ||
        isa_and_nonnull<linalg::BatchReduceMatmulOp>(linalgOp)))
    return false;

  auto aType =
      linalgOp.getDpsInputOperands()[0]->get().getType().cast<ShapedType>();
  auto bType =
      linalgOp.getDpsInputOperands()[1]->get().getType().cast<ShapedType>();
  auto cType =
      linalgOp.getDpsInitOperands()[0]->get().getType().cast<ShapedType>();

  ArrayRef<int64_t> shapeA = aType.getShape();
  ArrayRef<int64_t> shapeC = cType.getShape();
  int64_t m = shapeC[0];
  int64_t n = shapeC[1];
  // Matrix A might be 2D (gemm) or 3D (brgemm) but the last dimension will
  // always be reduction.
  int64_t k = shapeA.back();

  // For now, only M-N-K F16[16] x F16[16] x F16[16] WMMA variant is supported.
  // TODO: add more WMMA combinations.
  return aType.getElementType().isF16() && bType.getElementType().isF16() &&
         cType.getElementType().isF16() && m == 16 && n == 16 && k == 16;
}

// Create WMMA instructions out of matmul-like operation.
void gemmToGpuMMA(linalg::LinalgOp linalgOp, PatternRewriter &rewriter) {
  assert((isa_and_nonnull<linalg::MatmulOp>(linalgOp) ||
          isa_and_nonnull<linalg::BatchReduceMatmulOp>(linalgOp)) &&
         "Requires a matmul like op for mma lowering");

  Location loc = linalgOp.getLoc();

  auto matA = linalgOp.getDpsInputOperands()[0]->get();
  auto matB = linalgOp.getDpsInputOperands()[1]->get();
  auto matC = linalgOp.getDpsInitOperands()[0]->get();

  auto typeA = matA.getType().cast<ShapedType>();
  auto typeB = matB.getType().cast<ShapedType>();
  auto typeC = matC.getType().cast<ShapedType>();

  gpu::MMAMatrixType mmaTypeA = gpu::MMAMatrixType::get(
      typeA.getShape().take_back(2), typeA.getElementType(), "AOp");
  gpu::MMAMatrixType mmaTypeB = gpu::MMAMatrixType::get(
      typeB.getShape().take_back(2), typeB.getElementType(), "BOp");
  gpu::MMAMatrixType mmaTypeC =
      gpu::MMAMatrixType::get(typeC.getShape(), typeC.getElementType(), "COp");

  // Matrix A might be 2D (gemm) or 3D (brgemm) but the last dimension will
  // always be reduction.
  auto ldb = rewriter.getIndexAttr(typeA.getShape().back());
  auto lda = rewriter.getIndexAttr(typeC.getDimSize(0));
  auto ldc = rewriter.getIndexAttr(typeC.getDimSize(0));

  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

  // Fetch the inital value of the output element.
  Value tileC = rewriter
                    .create<gpu::SubgroupMmaLoadMatrixOp>(
                        loc, mmaTypeC, matC, ValueRange{zero, zero}, ldc,
                        /*transpose=*/UnitAttr())
                    .getRes();

  OpBuilder::InsertionGuard guard(rewriter);

  bool isBrgemm = isa<linalg::BatchReduceMatmulOp>(linalgOp);
  scf::ForOp batchLoop;
  Value batchIv;
  if (isBrgemm) {
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value batch =
        rewriter.create<arith::ConstantIndexOp>(loc, typeA.getDimSize(0));
    batchLoop =
        rewriter.create<scf::ForOp>(loc, zero, batch, one, ValueRange{tileC});
    rewriter.setInsertionPointToStart(batchLoop.getBody());
    batchIv = batchLoop.getInductionVar();
    tileC = batchLoop.getRegionIterArg(0);
  }

  auto readIndices =
      isBrgemm ? ValueRange{batchIv, zero, zero} : ValueRange{zero, zero};

  Value tileA =
      rewriter
          .create<gpu::SubgroupMmaLoadMatrixOp>(
              loc, mmaTypeA, matA, readIndices, lda, /*transpose=*/UnitAttr())
          .getRes();
  Value tileB =
      rewriter
          .create<gpu::SubgroupMmaLoadMatrixOp>(
              loc, mmaTypeB, matB, readIndices, ldb, /*transpose=*/UnitAttr())
          .getRes();

  Value result = rewriter
                     .create<gpu::SubgroupMmaComputeOp>(
                         loc, tileA, tileB, tileC, /*a_transpose=*/UnitAttr(),
                         /*b_transpose=*/UnitAttr())
                     .getRes();

  if (isBrgemm) {
    rewriter.setInsertionPointToEnd(batchLoop.getBody());
    rewriter.create<scf::YieldOp>(loc, ValueRange{result});
    result = batchLoop.getResults()[0];
    rewriter.setInsertionPointAfter(batchLoop);
  }

  // Write back the total sum to the output buffer.
  rewriter.create<gpu::SubgroupMmaStoreMatrixOp>(
      loc, result, matC, ValueRange{zero, zero}, ldc, /*transpose=*/UnitAttr());
}

// Create loops out of matmul-like operation.
void gemmToGpuLoops(linalg::LinalgOp linalgOp, PatternRewriter &rewriter) {
  assert((isa_and_nonnull<linalg::MatmulOp>(linalgOp) ||
          isa_and_nonnull<linalg::BatchReduceMatmulOp>(linalgOp)) &&
         "Requires a matmul like op for loop lowering");

  Location loc = linalgOp.getLoc();

  auto matA = linalgOp.getDpsInputOperands()[0]->get();
  auto matB = linalgOp.getDpsInputOperands()[1]->get();
  auto matC = linalgOp.getDpsInitOperands()[0]->get();

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
  rewriter.create<memref::StoreOp>(loc, result, matC, parallelIvs);
}

// Convert linalg.matmul to GPU-compatible kernel.
struct ConvertGemmToGpu : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  ConvertGemmToGpu(MLIRContext *ctx, bool useWmma)
      : OpRewritePattern(ctx), useWmma(useWmma) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    if (!matmulOp.hasBufferSemantics()) {
      return rewriter.notifyMatchFailure(
          matmulOp, "Linalg gemm to GPU expects memref type");
    }

    if (useWmma && supportsMMACompute(matmulOp))
      gemmToGpuMMA(matmulOp, rewriter);
    else
      gemmToGpuLoops(matmulOp, rewriter);

    rewriter.eraseOp(matmulOp);
    return success();
  }

private:
  bool useWmma;
};

// Convert linalg.batch_reduce_matmul to GPU-compatible kernel.
struct ConvertBrgemmToGpu
    : public OpRewritePattern<linalg::BatchReduceMatmulOp> {
  using OpRewritePattern<linalg::BatchReduceMatmulOp>::OpRewritePattern;

  ConvertBrgemmToGpu(MLIRContext *ctx, bool useWmma)
      : OpRewritePattern(ctx), useWmma(useWmma) {}

  LogicalResult matchAndRewrite(linalg::BatchReduceMatmulOp brgemmOp,
                                PatternRewriter &rewriter) const override {
    if (!brgemmOp.hasBufferSemantics()) {
      return rewriter.notifyMatchFailure(
          brgemmOp, "Linalg brgemm to GPU expects memref type");
    }

    if (useWmma && supportsMMACompute(brgemmOp))
      gemmToGpuMMA(brgemmOp, rewriter);
    else
      gemmToGpuLoops(brgemmOp, rewriter);

    rewriter.eraseOp(brgemmOp);
    return success();
  }

private:
  bool useWmma;
};

void populateLinalgToGpuPatterns(RewritePatternSet &patterns, bool useWmma) {
  patterns.add<ConvertGemmToGpu, ConvertBrgemmToGpu>(patterns.getContext(),
                                                     useWmma);
}

struct LinalgToGpu : public LinalgToGpuBase<LinalgToGpu> {
  LinalgToGpu() = default;
  LinalgToGpu(bool useWmma) { this->useWmma = useWmma; }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateLinalgToGpuPatterns(patterns, useWmma);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createLinalgToGpuPass(bool useWmma) {
  return std::make_unique<LinalgToGpu>(useWmma);
}
