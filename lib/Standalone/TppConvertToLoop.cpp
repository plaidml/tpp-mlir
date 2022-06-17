//===- ConvertTppToLoops.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/Tpp/TppOps.h"
#include "Standalone/TppPasses.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "Standalone/TppPasses.h.inc"

namespace {

//
// tpp.add ins(%a, %b) out(%c)
//
// Converts to:
//
// scf.some_loop(%i, %j)
//   %0 = load from %a
//   %1 = load from %b
//   %2 = add %0, %1
//   store %2 to %c
//
struct ConvertTppAddOp : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp addOp,
                                PatternRewriter &rewriter) const override {
    Location loc = addOp.getLoc();
    SmallVector<Value> ubs;
    size_t rank = addOp.lhs().getType().cast<MemRefType>().getRank();
    for (size_t idx = 0; idx < rank; idx++) {
      Value dim = rewriter.create<arith::ConstantIndexOp>(
          loc, addOp.lhs().getType().cast<MemRefType>().getShape()[idx]);
      ubs.push_back(dim);
    }
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> lbs(rank, zero);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<Value> steps(rank, one);
    (void)scf::buildLoopNest(
        rewriter, loc, lbs, ubs, steps,
        [&](OpBuilder &b, Location loc, ValueRange localIvs) {
          Value scalarLhs =
              b.create<memref::LoadOp>(loc, addOp.lhs(), localIvs);
          Value scalarRhs =
              b.create<memref::LoadOp>(loc, addOp.rhs(), localIvs);
          Value addLhsAndRhs =
              b.create<arith::AddFOp>(loc, scalarLhs, scalarRhs);
          b.create<memref::StoreOp>(loc, addLhsAndRhs, addOp.output(),
                                    localIvs);
        });
    rewriter.eraseOp(addOp);
    return success();
  }
};

// Lowers identity op.
struct ConvertTppIdentityOp : public OpRewritePattern<IdentityOp> {
  using OpRewritePattern<IdentityOp>::OpRewritePattern;

  bool isScalar(Value val) const { return !val.getType().isa<ShapedType>(); }

  bool is1DMemRef(Value val) const {
    if (isScalar(val))
      return false;
    return val.getType().cast<ShapedType>().getRank() == 1;
  }

  LogicalResult matchAndRewrite(IdentityOp identityOp,
                                PatternRewriter &rewriter) const override {
    Location loc = identityOp.getLoc();
    // Build loop nests.
    SmallVector<Value> ubs;
    size_t rank = identityOp.output().getType().cast<MemRefType>().getRank();
    for (size_t idx = 0; idx < rank; idx++) {
      Value dim = rewriter.create<arith::ConstantIndexOp>(
          loc,
          identityOp.output().getType().cast<MemRefType>().getShape()[idx]);
      ubs.push_back(dim);
    }
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> lbs(rank, zero);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<Value> steps(rank, one);
    (void)scf::buildLoopNest(
        rewriter, loc, lbs, ubs, steps,
        [&](OpBuilder &b, Location loc, ValueRange localIvs) {
          Value input = identityOp.input();
          // input is scalar.
          if (isScalar(input))
            b.create<memref::StoreOp>(loc, input, identityOp.output(),
                                      localIvs);
          // input is a 1d-memref.
          else if (is1DMemRef(input)) {
            Value scalarVal = b.create<memref::LoadOp>(loc, input, localIvs[1]);
            b.create<memref::StoreOp>(loc, scalarVal, identityOp.output(),
                                      localIvs);
          }
          // input is a 2d-memref.
          else {
            Value scalarVal = b.create<memref::LoadOp>(loc, input, localIvs);
            b.create<memref::StoreOp>(loc, scalarVal, identityOp.output(),
                                      localIvs);
          }
        });
    rewriter.eraseOp(identityOp);
    return success();
  }
};

struct ConvertTppReluOp : public OpRewritePattern<ReluOp> {
  using OpRewritePattern<ReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReluOp reluOp,
                                PatternRewriter &rewriter) const override {
    Location loc = reluOp.getLoc();
    SmallVector<Value> ubs;
    size_t rank = reluOp.input().getType().cast<MemRefType>().getRank();
    for (size_t idx = 0; idx < rank; idx++) {
      Value dim = rewriter.create<arith::ConstantIndexOp>(
          loc, reluOp.input().getType().cast<MemRefType>().getShape()[idx]);
      ubs.push_back(dim);
    }
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> lbs(rank, zero);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<Value> steps(rank, one);

    Type elementType =
        reluOp.input().getType().cast<MemRefType>().getElementType();
    Value zeroConstant = rewriter.create<arith::ConstantOp>(
        loc, elementType, rewriter.getFloatAttr(elementType, 0));

    (void)scf::buildLoopNest(
        rewriter, loc, lbs, ubs, steps,
        [&](OpBuilder &b, Location loc, ValueRange localIvs) {
          Value scalarLhs =
              b.create<memref::LoadOp>(loc, reluOp.input(), localIvs);
          Value scalarRelu =
              b.create<arith::MaxFOp>(loc, zeroConstant, scalarLhs);
          b.create<memref::StoreOp>(loc, scalarRelu, reluOp.output(), localIvs);
        });

    rewriter.eraseOp(reluOp);
    return success();
  }
};

struct ConvertTppMatmulOp : public OpRewritePattern<MatmulOp> {
  using OpRewritePattern<MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    Location loc = matmulOp.getLoc();
    ArrayRef<int64_t> shapeC =
        matmulOp.matrixC().getType().cast<MemRefType>().getShape();
    ArrayRef<int64_t> shapeA =
        matmulOp.matrixA().getType().cast<MemRefType>().getShape();
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
          Value scalarA = b.create<memref::LoadOp>(loc, matmulOp.matrixA(),
                                                   ValueRange{localI, localK});
          Value scalarB = b.create<memref::LoadOp>(loc, matmulOp.matrixB(),
                                                   ValueRange{localK, localJ});
          Value scalarC = b.create<memref::LoadOp>(loc, matmulOp.matrixC(),
                                                   ValueRange{localI, localJ});
          Value scalarMul = b.create<arith::MulFOp>(loc, scalarA, scalarB);
          Value scalarAdd = b.create<arith::AddFOp>(loc, scalarC, scalarMul);
          b.create<memref::StoreOp>(loc, scalarAdd, matmulOp.matrixC(),
                                    ValueRange{localI, localJ});
        });

    rewriter.eraseOp(matmulOp);
    return success();
  }
};

void populateTppToLoopsPatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<ConvertTppAddOp, 
               ConvertTppIdentityOp,
               ConvertTppMatmulOp,
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

std::unique_ptr<OperationPass<func::FuncOp>> mlir::tpp::createTppToLoopsPass() {
  return std::make_unique<ConvertTppToLoops>();
}
