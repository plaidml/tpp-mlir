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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

// Convert linalg.matmul to GPU-compatible kernel.
struct ConvertMatmulToGpu : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    if (!matmulOp.hasBufferSemantics())
      return rewriter.notifyMatchFailure(matmulOp,
                                         "Linalg to GPU expects memref type");

    Location loc = matmulOp.getLoc();
    ArrayRef<int64_t> shapeC = matmulOp.getDpsInitOperand(0)
                                   ->get()
                                   .getType()
                                   .cast<ShapedType>()
                                   .getShape();
    ArrayRef<int64_t> shapeB =
        matmulOp.getInputs()[1].getType().cast<ShapedType>().getShape();
    ArrayRef<int64_t> shapeA =
        matmulOp.getInputs()[0].getType().cast<ShapedType>().getShape();
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

    auto parallelLoop = rewriter.create<scf::ParallelOp>(
        loc, ValueRange{zero, zero}, ValueRange{i, j}, ValueRange{one, one});
    auto parallelIvs = parallelLoop.getInductionVars();
    ivs.append(parallelIvs.begin(), parallelIvs.end());

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(parallelLoop.getBody()->getTerminator());

    auto outputBuf = matmulOp.getDpsInitOperand(0)->get();
    Value initVal = rewriter.create<memref::LoadOp>(loc, outputBuf, parallelIvs)
                        .getResult();

    auto bodyBuilder = [&](OpBuilder &b, Location loc, Value localIv,
                           ValueRange iterArgs) {
      SmallVector<Value> loopIvs = ivs;
      loopIvs.push_back(localIv);
      assert(loopIvs.size() == 3);
      Value localI = loopIvs[0];
      Value localJ = loopIvs[1];
      Value localK = loopIvs[2];
      Value scalarA = b.create<memref::LoadOp>(loc, matmulOp.getInputs()[0],
                                               ValueRange{localI, localK});
      Value scalarB = b.create<memref::LoadOp>(loc, matmulOp.getInputs()[1],
                                               ValueRange{localK, localJ});
      Value scalarMul = b.create<arith::MulFOp>(loc, scalarA, scalarB);
      auto scalarAdd = b.create<arith::AddFOp>(loc, iterArgs[0], scalarMul);

      b.create<scf::YieldOp>(loc, scalarAdd.getResult());
    };

    auto accumulationLoop = rewriter.create<scf::ForOp>(
        loc, zero, k, one, ValueRange{initVal},
        [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
          bodyBuilder(b, loc, iv, iterArgs);
        });

    rewriter.create<memref::StoreOp>(loc, accumulationLoop.getResults()[0],
                                     outputBuf, parallelIvs);

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
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createLinalgToGpuPass() {
  return std::make_unique<LinalgToGpu>();
}
