//===- GpuConversion.cpp -----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

// Convert linalg.matmul to GPU-compatible kernel.
struct ConvertMatmulToGpu : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern<FusedBrgemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    if (!matmulOp.hasBufferSemantics())
      return rewriter.notifyMatchFailure(fusedBrgemmOp,
                                         "Linalg to GPU expects memref type");

    Location loc = matmulOp.getLoc();
    ArrayRef<int64_t> shapeC = matmulOp.getOutputType().getShape();
    ArrayRef<int64_t> shapeB = matmulOp.getMemRefInputType(1).getShape();
    ArrayRef<int64_t> shapeA = matmulOp.getMemRefInputType(0).getShape();
    if (shapeB.size() == 3) {
      return rewriter.notifyMatchFailure(matmulOp,
                                         "Packed BF16 loops unsupported");
    }
    // Parallel dims.
    Value i = rewriter.create<arith::ConstantIndexOp>(loc, shapeC[0]);
    Value j = rewriter.create<arith::ConstantIndexOp>(loc, shapeC[1]);
    // Reduction dim.
    Value k = rewriter.create<arith::ConstantIndexOp>(loc, shapeA[1]);
    SmallVector<Value> ubs = {i, j, k};
    // Lbs.
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> lbs = {zero, zero, zero};
    // Step.
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<Value> steps = {one, one, one};
    SmallVector<Value> ivs;

    Value initVal = b.create<memref::LoadOp>(loc, matmulOp.getInputs()[2],
                                             ValueRange{localI, localJ});

    auto bodyBuilder = [&](OpBuilder &b, Location loc, ValueRange localIvs) {
      SmallVector<Value> loopIvs = ivs;
      loopIvs.append(localIvs.begin(), localIvs.end());
      assert(loopIvs.size() == 3);
      Value localI = loopIvs[0];
      Value localJ = loopIvs[1];
      Value localK = loopIvs[2];
      Value scalarA = b.create<memref::LoadOp>(loc, matmulOp.getInputs()[0],
                                               ValueRange{localI, localK});
      Value scalarB = b.create<memref::LoadOp>(loc, matmulOp.getInputs()[1],
                                               ValueRange{localK, localJ});
      Value scalarC = b.create<memref::LoadOp>(loc, matmulOp.getInputs()[2],
                                               ValueRange{localI, localJ});
      Value scalarMul = b.create<arith::MulFOp>(loc, scalarA, scalarB);
      Value scalarAdd = b.create<arith::AddFOp>(loc, scalarC, scalarMul);
      b.create<memref::StoreOp>(loc, scalarAdd, matmulOp.getOutput(),
                                ValueRange{localI, localJ});
    };

    auto parallelLoop = rewriter.create<scf::ParallelOp>(
        loc, ValueRange{zero, zero}, ValueRange{i, j}, ValueRange{one, one});
    auto parallelIvs = parallelLoop.getInductionVars();
    ivs.append(parallelIvs.begin(), parallelIvs.end());

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(parallelLoop.getBody()->getTerminator());
    rewriter.create<scf::ForOp>(
        loc, zero, k, one, std::nullopt,
        [&](OpBuilder &b, Location loc, Value iv, ValueRange args) {
          bodyBuilder(b, loc, iv);
          b.create<scf::YieldOp>(loc);
        });

    rewriter.eraseOp(matmulOp);
    return success();
  }
};

void populateLinalgToGpuPatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<ConvertMatmulToGpu>(patterns.getContext());
  // clang-format on
}

struct LinalgToGpu : public LinalgToGpuBase<LinalgToGpu> {
  LinalgToGpu() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<gpu::GPUDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateLinalgToGpuPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
}

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createLinalgToGpuPass() {
  return std::make_unique<LinalgToGpu>();
}
