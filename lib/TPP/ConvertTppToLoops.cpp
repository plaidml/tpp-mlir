//===- ConvertTppToLoops.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Tpp/TppOps.h"
#include "TPP/Dialect/Tpp/TppUtils.h"
#include "TPP/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

// Convert tpp.add to SCF loops.
struct ConvertTppAddOp : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  ConvertTppAddOp(MLIRContext *ctx, bool parallel)
      : OpRewritePattern(ctx), parallel(parallel) {}

  LogicalResult matchAndRewrite(AddOp addOp,
                                PatternRewriter &rewriter) const override {
    if (!addOp.hasBufferSemantics())
      return rewriter.notifyMatchFailure(
          addOp, "Tpp loop lowering expects memref type");

    Location loc = addOp.getLoc();

    SmallVector<Value> ubs;
    size_t rank = addOp.getInputs()[0].getType().cast<MemRefType>().getRank();
    for (size_t idx = 0; idx < rank; idx++) {
      Value dim = rewriter.create<arith::ConstantIndexOp>(
          loc,
          addOp.getInputs()[0].getType().cast<MemRefType>().getShape()[idx]);
      ubs.push_back(dim);
    }
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> lbs(rank, zero);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<Value> steps(rank, one);

    auto bodyBuilder = [&](OpBuilder &b, Location loc, ValueRange localIvs) {
      Value scalarLhs =
          b.create<memref::LoadOp>(loc, addOp.getInputs()[0], localIvs);
      Value scalarRhs =
          b.create<memref::LoadOp>(loc, addOp.getInputs()[1], localIvs);
      Value addLhsAndRhs = b.create<arith::AddFOp>(loc, scalarLhs, scalarRhs);
      b.create<memref::StoreOp>(loc, addLhsAndRhs, addOp.getOutput(), localIvs);
    };

    if (parallel)
      rewriter.create<scf::ParallelOp>(loc, lbs, ubs, steps, bodyBuilder);
    else
      (void)scf::buildLoopNest(rewriter, loc, lbs, ubs, steps, bodyBuilder);

    rewriter.eraseOp(addOp);
    return success();
  }

private:
  bool parallel;
};

void buildUnaryLoop(
    PatternRewriter &rewriter, TppOp tppOp,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilder,
    bool parallel = false) {
  assert(tppOp.isUnary() && "Expect tpp unary op");

  Location loc = tppOp.getLoc();

  SmallVector<Value> ubs;
  size_t rank = tppOp.getOutput().getType().cast<MemRefType>().getRank();
  ArrayRef<int64_t> shapeOutput =
      tppOp.getOutput().getType().cast<MemRefType>().getShape();
  for (size_t idx = 0; idx < rank; idx++) {
    Value dim = rewriter.create<arith::ConstantIndexOp>(loc, shapeOutput[idx]);
    ubs.push_back(dim);
  }

  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> lbs(rank, zero);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Value> steps(rank, one);

  if (parallel)
    rewriter.create<scf::ParallelOp>(loc, lbs, ubs, steps, bodyBuilder);
  else
    (void)scf::buildLoopNest(rewriter, loc, lbs, ubs, steps, bodyBuilder);
}

// Converts tpp.identity to SCF loops.
struct ConvertTppIdentityOp : public OpRewritePattern<IdentityOp> {
  using OpRewritePattern<IdentityOp>::OpRewritePattern;

  ConvertTppIdentityOp(MLIRContext *ctx, bool parallel)
      : OpRewritePattern(ctx), parallel(parallel) {}

  bool isScalar(Value val) const { return !val.getType().isa<ShapedType>(); }

  bool is1DMemRef(Value val) const {
    if (isScalar(val))
      return false;
    return val.getType().cast<ShapedType>().getRank() == 1;
  }

  LogicalResult matchAndRewrite(IdentityOp identityOp,
                                PatternRewriter &rewriter) const override {
    if (!identityOp.hasBufferSemantics())
      return rewriter.notifyMatchFailure(
          identityOp, "Tpp loop lowering expects memref type");

    auto bodyBuilder = [&](OpBuilder &b, Location loc, ValueRange localIvs) {
      Value input = identityOp.getInputs()[0];
      // input is scalar.
      if (isScalar(input))
        b.create<memref::StoreOp>(loc, input, identityOp.getOutput(), localIvs);
      // input is a 1d-memref.
      else if (is1DMemRef(input)) {
        Value scalarVal = b.create<memref::LoadOp>(loc, input, localIvs[1]);
        b.create<memref::StoreOp>(loc, scalarVal, identityOp.getOutput(),
                                  localIvs);
      }
      // input is a 2d-memref.
      else {
        ArrayRef<int64_t> shapeInput =
            identityOp.getInputs()[0].getType().cast<MemRefType>().getShape();
        SmallVector<Value, 2> inputIvs = localIvs;
        // broadcasting dimension with size 1.
        Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        for (size_t idx = 0; idx < shapeInput.size(); idx++)
          if (shapeInput[idx] == 1)
            inputIvs[idx] = zero;
        Value scalarVal = b.create<memref::LoadOp>(loc, input, inputIvs);
        b.create<memref::StoreOp>(loc, scalarVal, identityOp.getOutput(),
                                  localIvs);
      }
    };
    buildUnaryLoop(rewriter, identityOp, bodyBuilder, parallel);

    rewriter.eraseOp(identityOp);
    return success();
  }

private:
  bool parallel;
};

// Convert tpp.relu to SCF loops.
struct ConvertTppReluOp : public OpRewritePattern<ReluOp> {
  using OpRewritePattern<ReluOp>::OpRewritePattern;

  ConvertTppReluOp(MLIRContext *ctx, bool parallel)
      : OpRewritePattern(ctx), parallel(parallel) {}

  LogicalResult matchAndRewrite(ReluOp reluOp,
                                PatternRewriter &rewriter) const override {
    if (!reluOp.hasBufferSemantics())
      return rewriter.notifyMatchFailure(
          reluOp, "Tpp loop lowering expects memref type");

    Location loc = reluOp.getLoc();
    Type elementType =
        reluOp.getOutput().getType().cast<MemRefType>().getElementType();
    Value zeroConstant = rewriter.create<arith::ConstantOp>(
        loc, elementType, rewriter.getFloatAttr(elementType, 0));

    auto bodyBuilder = [&](OpBuilder &b, Location loc, ValueRange localIvs) {
      Value scalarLhs =
          b.create<memref::LoadOp>(loc, reluOp.getInputs()[0], localIvs);
      Value scalarRelu =
          b.create<arith::MaximumFOp>(loc, zeroConstant, scalarLhs);
      b.create<memref::StoreOp>(loc, scalarRelu, reluOp.getOutput(), localIvs);
    };
    buildUnaryLoop(rewriter, reluOp, bodyBuilder, parallel);

    rewriter.eraseOp(reluOp);
    return success();
  }

private:
  bool parallel;
};

// Convert tpp.zero to SCF loops.
struct ConvertTppZeroOp : public OpRewritePattern<ZeroOp> {
  using OpRewritePattern<ZeroOp>::OpRewritePattern;

  ConvertTppZeroOp(MLIRContext *ctx, bool parallel)
      : OpRewritePattern(ctx), parallel(parallel) {}

  LogicalResult matchAndRewrite(ZeroOp zeroOp,
                                PatternRewriter &rewriter) const override {
    if (!zeroOp.hasBufferSemantics())
      return rewriter.notifyMatchFailure(
          zeroOp, "Tpp loop lowering expects memref type");

    Location loc = zeroOp.getLoc();

    Type elementType =
        zeroOp.getOutput().getType().cast<MemRefType>().getElementType();
    Value zeroConstant = rewriter.create<arith::ConstantOp>(
        loc, elementType, rewriter.getFloatAttr(elementType, 0));

    // handle memref case.
    auto bodyBuilder = [&](OpBuilder &b, Location loc, ValueRange localIvs) {
      b.create<memref::StoreOp>(loc, zeroConstant, zeroOp.getOutput(),
                                localIvs);
    };
    buildUnaryLoop(rewriter, zeroOp, bodyBuilder, parallel);

    rewriter.eraseOp(zeroOp);
    return success();
  }

private:
  bool parallel;
};

// Convert tpp.gemm to SCF loops.
struct ConvertTppGemmOp : public OpRewritePattern<GemmOp> {
  using OpRewritePattern<GemmOp>::OpRewritePattern;

  ConvertTppGemmOp(MLIRContext *ctx, bool parallel)
      : OpRewritePattern(ctx), parallel(parallel) {}

  LogicalResult matchAndRewrite(GemmOp matmulOp,
                                PatternRewriter &rewriter) const override {
    if (!matmulOp.hasBufferSemantics())
      return rewriter.notifyMatchFailure(
          matmulOp, "Tpp loop lowering expects memref type");

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

    if (parallel) {
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
    } else
      (void)scf::buildLoopNest(rewriter, loc, lbs, ubs, steps, bodyBuilder);

    rewriter.eraseOp(matmulOp);
    return success();
  }

private:
  bool parallel;
};

// Convert tpp.brgemm to SCF loops.
struct ConvertTppBrgemmOp : public OpRewritePattern<BrgemmOp> {
  using OpRewritePattern<BrgemmOp>::OpRewritePattern;

  ConvertTppBrgemmOp(MLIRContext *ctx, bool parallel)
      : OpRewritePattern(ctx), parallel(parallel) {}

  LogicalResult matchAndRewrite(BrgemmOp brgemmOp,
                                PatternRewriter &rewriter) const override {
    if (!brgemmOp.hasBufferSemantics())
      return rewriter.notifyMatchFailure(
          brgemmOp, "Tpp loop lowering expects memref type");

    Location loc = brgemmOp.getLoc();
    ArrayRef<int64_t> shapeC = brgemmOp.getOutputType().getShape();
    ArrayRef<int64_t> shapeA = brgemmOp.getMemRefInputType(0).getShape();
    // Parallel dims.
    Value i = rewriter.createOrFold<arith::ConstantIndexOp>(loc, shapeC[0]);
    Value j = rewriter.createOrFold<arith::ConstantIndexOp>(loc, shapeC[1]);
    // Reduction dims.
    Value k = rewriter.createOrFold<arith::ConstantIndexOp>(loc, shapeA[2]);
    Value b = rewriter.createOrFold<arith::ConstantIndexOp>(loc, shapeA[0]);
    SmallVector<Value> ubs = {b, i, j, k};
    // Lbs.
    Value zero = rewriter.createOrFold<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> lbs = {zero, zero, zero, zero};
    // Step.
    Value one = rewriter.createOrFold<arith::ConstantIndexOp>(loc, 1);
    SmallVector<Value> steps = {one, one, one, one};
    SmallVector<Value> ivs;

    auto bodyBuilder = [&](OpBuilder &b, Location loc, ValueRange localIvs) {
      SmallVector<Value> loopIvs = ivs;
      loopIvs.append(localIvs.begin(), localIvs.end());
      assert(loopIvs.size() == 4);
      Value localB = loopIvs[0];
      Value localI = loopIvs[1];
      Value localJ = loopIvs[2];
      Value localK = loopIvs[3];
      Value scalarA = b.create<memref::LoadOp>(
          loc, brgemmOp.getInputs()[0], ValueRange{localB, localI, localK});
      Value scalarB = b.create<memref::LoadOp>(
          loc, brgemmOp.getInputs()[1], ValueRange{localB, localK, localJ});
      Value scalarC = b.create<memref::LoadOp>(loc, brgemmOp.getInputs()[2],
                                               ValueRange{localI, localJ});
      Value scalarMul = b.create<arith::MulFOp>(loc, scalarA, scalarB);
      Value scalarAdd = b.create<arith::AddFOp>(loc, scalarC, scalarMul);
      b.create<memref::StoreOp>(loc, scalarAdd, brgemmOp.getOutput(),
                                ValueRange{localI, localJ});
    };

    if (parallel) {
      auto parallelLoop = rewriter.create<scf::ParallelOp>(
          loc, ValueRange{zero, zero}, ValueRange{i, j}, ValueRange{one, one});
      auto parallelIvs = parallelLoop.getInductionVars();

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(parallelLoop.getBody()->getTerminator());
      auto batchLoop = rewriter.create<scf::ForOp>(loc, zero, b, one);

      // Keep IVs in the original order: b, i, j, k.
      ivs.push_back(batchLoop.getInductionVar());
      ivs.append(parallelIvs.begin(), parallelIvs.end());

      rewriter.setInsertionPoint(batchLoop.getBody()->getTerminator());
      rewriter.create<scf::ForOp>(
          loc, zero, k, one, std::nullopt,
          [&](OpBuilder &b, Location loc, Value iv, ValueRange args) {
            bodyBuilder(b, loc, iv);
            b.create<scf::YieldOp>(loc);
          });
    } else
      (void)scf::buildLoopNest(rewriter, loc, lbs, ubs, steps, bodyBuilder);

    rewriter.eraseOp(brgemmOp);
    return success();
  }

private:
  bool parallel;
};

// Convert tpp.fused_brgemm to SCF loops.
struct ConvertTppFusedBrgemmOp : public OpRewritePattern<FusedBrgemmOp> {
  using OpRewritePattern<FusedBrgemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FusedBrgemmOp fusedBrgemmOp,
                                PatternRewriter &rewriter) const override {
    if (!fusedBrgemmOp.hasBufferSemantics())
      return rewriter.notifyMatchFailure(
          fusedBrgemmOp, "Tpp loop lowering expects memref type");

    return tpp::utils::splitAndReplaceFusedOp(fusedBrgemmOp, rewriter);
  }
};

void populateTppToLoopsPatterns(RewritePatternSet &patterns, bool parallel) {
  // clang-format off
  patterns.add<ConvertTppAddOp, 
               ConvertTppIdentityOp,
               ConvertTppGemmOp,
               ConvertTppBrgemmOp,
               ConvertTppFusedBrgemmOp,
               ConvertTppReluOp,
               ConvertTppZeroOp>(patterns.getContext(), parallel);
  // clang-format on
}

struct ConvertTppToLoops : public ConvertTppToLoopsBase<ConvertTppToLoops> {
  ConvertTppToLoops() = default;
  ConvertTppToLoops(bool parallel) { this->parallel = parallel; }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateTppToLoopsPatterns(patterns, parallel);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createConvertTppToLoopsPass(bool parallel) {
  return std::make_unique<ConvertTppToLoops>(parallel);
}
