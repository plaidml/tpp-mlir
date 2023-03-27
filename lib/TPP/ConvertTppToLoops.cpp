//===- ConvertTppToLoops.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Tpp/TppOps.h"
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

static bool isScalarVal(Value val) { return !val.getType().isa<ShapedType>(); }

static SmallVector<Value> getLocalIvs(Value in, Value out, ValueRange ivs,
                                      Value zero) {
  assert(in.getType().isa<ShapedType>());
  assert(out.getType().isa<ShapedType>());
  auto shapedIn = in.getType().dyn_cast<ShapedType>();
  auto shapedOut = out.getType().dyn_cast<ShapedType>();
  assert(shapedOut.getRank() >= shapedIn.getRank());

  // Handle rank 1 input.
  auto shapeIn = shapedIn.getShape();
  if (shapedIn.getRank() == 1) {
    if (shapeIn[0] == 1)
      return {zero};
    return {ivs[shapedOut.getRank() - 1]};
  }

  // Handle rank 2 input.
  assert(shapedIn.getRank() == shapedOut.getRank());
  SmallVector<Value> newIvs;
  for (auto [idx, shapeOnDim] : llvm::enumerate(shapeIn)) {
    if (shapeOnDim == 1)
      newIvs.push_back(zero);
    else
      newIvs.push_back(ivs[idx]);
  }
  return newIvs;
}

static inline SmallVector<Value> getShapeAsValues(RewriterBase &rewriter,
                                                  Location loc,
                                                  ArrayRef<int64_t> shape) {
  SmallVector<Value> shapeAsValues;
  for (int64_t shapeDim : shape) {
    assert(shapeDim != ShapedType::kDynamic);
    shapeAsValues.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, shapeDim));
  }
  return shapeAsValues;
}

static inline Value getZero(RewriterBase &rewriter, Location loc) {
  return rewriter.create<arith::ConstantIndexOp>(loc, 0);
}

static inline Value getOne(RewriterBase &rewriter, Location loc) {
  return rewriter.create<arith::ConstantIndexOp>(loc, 1);
}

template <typename OpTy>
static void convertTppToLoops(RewriterBase &rewriter, Location loc, Value lhs,
                              Value rhs, Value out) {
  assert(out.getType().isa<ShapedType>());
  auto shape = out.getType().cast<ShapedType>().getShape();
  int64_t rank = shape.size();
  SmallVector<Value> ubs;
  for (int64_t shapeDim : shape)
    ubs.push_back(rewriter.create<arith::ConstantIndexOp>(loc, shapeDim));
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> lbs(rank, zero);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Value> steps(rank, one);
  (void)scf::buildLoopNest(
      rewriter, loc, lbs, ubs, steps,
      [&](OpBuilder &b, Location loc, ValueRange localIvs) {
        Value scalarLhs =
            isScalarVal(lhs)
                ? lhs
                : b.create<memref::LoadOp>(
                      loc, lhs, getLocalIvs(lhs, out, localIvs, zero));
        Value scalarRhs =
            isScalarVal(rhs)
                ? rhs
                : b.create<memref::LoadOp>(
                      loc, rhs, getLocalIvs(rhs, out, localIvs, zero));
        Value opLhsAndRhs = b.create<OpTy>(loc, scalarLhs, scalarRhs);
        b.create<memref::StoreOp>(loc, opLhsAndRhs, out, localIvs);
      });
}

// Convert tpp.add to SCF loops.
struct ConvertTppAddOp : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp addOp,
                                PatternRewriter &rewriter) const override {
    convertTppToLoops<arith::AddFOp>(rewriter, addOp.getLoc(), addOp.getInputs()[0],
                                     addOp.getInputs()[1], addOp.getOutput());
    rewriter.eraseOp(addOp);
    return success();
  }
};

// Converts tpp.identity to SCF loops.
struct ConvertTppIdentityOp : public OpRewritePattern<IdentityOp> {
  using OpRewritePattern<IdentityOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IdentityOp identityOp,
                                PatternRewriter &rewriter) const override {

    Value out = identityOp.getOutput();
    Value in = identityOp.getInputs()[0];
    Location loc = identityOp.getLoc();
    assert(out.getType().isa<ShapedType>());
    auto shape = out.getType().cast<ShapedType>().getShape();
    int64_t rank = shape.size();
    Value zero = getZero(rewriter, loc);
    SmallVector<Value> ubs = getShapeAsValues(rewriter, loc, shape);
    SmallVector<Value> lbs(rank, zero);
    SmallVector<Value> steps(rank, getOne(rewriter, loc));

    (void)scf::buildLoopNest(
        rewriter, loc, lbs, ubs, steps,
        [&](OpBuilder &b, Location loc, ValueRange localIvs) {
          Value scalarIn =
              isScalarVal(in)
                  ? in
                  : b.create<memref::LoadOp>(
                        loc, in, getLocalIvs(in, out, localIvs, zero));
          b.create<memref::StoreOp>(loc, scalarIn, out, localIvs);
        });
    rewriter.eraseOp(identityOp);
    return success();
  }
};

// Convert tpp.relu to SCF loops.
struct ConvertTppReluOp : public OpRewritePattern<ReluOp> {
  using OpRewritePattern<ReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReluOp reluOp,
                                PatternRewriter &rewriter) const override {
    Location loc = reluOp.getLoc();
    Type elementType =
        reluOp.getOutput().getType().cast<MemRefType>().getElementType();
    Value zeroConstant = rewriter.create<arith::ConstantOp>(
        loc, elementType, rewriter.getFloatAttr(elementType, 0));
    convertTppToLoops<arith::MaxFOp>(rewriter, loc, reluOp.getInputs()[0],
                                     zeroConstant, reluOp.getOutput());
    rewriter.eraseOp(reluOp);
    return success();
  }
};

// Convert tpp.matmul to SCF loops.
struct ConvertTppMatmulOp : public OpRewritePattern<MatmulOp> {
  using OpRewritePattern<MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    Location loc = matmulOp.getLoc();
    ArrayRef<int64_t> shapeC = matmulOp.getOutputType().getShape();
    ArrayRef<int64_t> shapeB = matmulOp.getMemRefInputType(1).getShape();
    ArrayRef<int64_t> shapeA = matmulOp.getMemRefInputType(0).getShape();
    if (shapeB.size() == 3)
      return rewriter.notifyMatchFailure(matmulOp,
                                         "Packed BF16 loops unsupported");
    Value i = rewriter.create<arith::ConstantIndexOp>(loc, shapeC[0]);
    Value j = rewriter.create<arith::ConstantIndexOp>(loc, shapeC[1]);
    Value k = rewriter.create<arith::ConstantIndexOp>(loc, shapeA[1]);
    SmallVector<Value> ubs = {i, j, k};
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> lbs = {zero, zero, zero};
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<Value> steps = {one, one, one};

    (void)scf::buildLoopNest(
        rewriter, loc, lbs, ubs, steps,
        [&](OpBuilder &b, Location loc, ValueRange localIvs) {
          assert(localIvs.size() == 3);
          Value localI = localIvs[0];
          Value localJ = localIvs[1];
          Value localK = localIvs[2];
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
        });

    rewriter.eraseOp(matmulOp);
    return success();
  }
};

// Convert tpp.brgemm to SCF loops.
struct ConvertTppBrgemmOp : public OpRewritePattern<BrgemmOp> {
  using OpRewritePattern<BrgemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BrgemmOp brgemmOp,
                                PatternRewriter &rewriter) const override {
    Location loc = brgemmOp.getLoc();
    ArrayRef<int64_t> shapeC = brgemmOp.getOutputType().getShape();
    ArrayRef<int64_t> shapeA = brgemmOp.getMemRefInputType(0).getShape();
    Value i = rewriter.createOrFold<arith::ConstantIndexOp>(loc, shapeC[0]);
    Value j = rewriter.createOrFold<arith::ConstantIndexOp>(loc, shapeC[1]);
    Value k = rewriter.createOrFold<arith::ConstantIndexOp>(loc, shapeA[2]);
    Value b = rewriter.createOrFold<arith::ConstantIndexOp>(loc, shapeA[0]);
    SmallVector<Value> ubs = {b, i, j, k};
    Value zero = rewriter.createOrFold<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> lbs = {zero, zero, zero, zero};
    Value one = rewriter.createOrFold<arith::ConstantIndexOp>(loc, 1);
    SmallVector<Value> steps = {one, one, one, one};

    (void)scf::buildLoopNest(
        rewriter, loc, lbs, ubs, steps,
        [&](OpBuilder &b, Location loc, ValueRange localIvs) {
          assert(localIvs.size() == 4);
          Value localB = localIvs[0];
          Value localI = localIvs[1];
          Value localJ = localIvs[2];
          Value localK = localIvs[3];
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
        });
    rewriter.eraseOp(brgemmOp);
    return success();
  }
};

void populateTppToLoopsPatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<ConvertTppAddOp, 
               ConvertTppIdentityOp,
               ConvertTppMatmulOp,
               ConvertTppBrgemmOp,
               ConvertTppReluOp>(patterns.getContext());
  // clang-format on
}

struct ConvertTppToLoops : public ConvertTppToLoopsBase<ConvertTppToLoops> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateTppToLoopsPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createConvertTppToLoopsPass() {
  return std::make_unique<ConvertTppToLoops>();
}
