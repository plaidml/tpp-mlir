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

// Convert tpp.add to SCF loops.
struct ConvertTppAddOp : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  bool isScalarOp(AddOp addOp) const {
    return !addOp.getInputs()[0].getType().isa<ShapedType>();
  }

  LogicalResult matchAndRewrite(AddOp addOp,
                                PatternRewriter &rewriter) const override {
    Location loc = addOp.getLoc();
    // handle scalar case.
    if (isScalarOp(addOp)) {
      Value scalarAdd = rewriter.create<arith::AddFOp>(
          loc, addOp.getInputs()[0], addOp.getInputs()[1]);
      rewriter.replaceAllUsesWith(addOp.getOutput(), scalarAdd);
      rewriter.eraseOp(addOp);
      return success();
    }
    // handle memref case.
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
    (void)scf::buildLoopNest(
        rewriter, loc, lbs, ubs, steps,
        [&](OpBuilder &b, Location loc, ValueRange localIvs) {
          Value scalarLhs =
              b.create<memref::LoadOp>(loc, addOp.getInputs()[0], localIvs);
          Value scalarRhs =
              b.create<memref::LoadOp>(loc, addOp.getInputs()[1], localIvs);
          Value addLhsAndRhs =
              b.create<arith::AddFOp>(loc, scalarLhs, scalarRhs);
          b.create<memref::StoreOp>(loc, addLhsAndRhs, addOp.getOutput(),
                                    localIvs);
        });
    rewriter.eraseOp(addOp);
    return success();
  }
};

// Converts tpp.identity to SCF loops.
struct ConvertTppIdentityOp : public OpRewritePattern<IdentityOp> {
  using OpRewritePattern<IdentityOp>::OpRewritePattern;

  bool isScalar(Value val) const { return !val.getType().isa<ShapedType>(); }

  bool is1DMemRef(Value val) const {
    if (isScalar(val))
      return false;
    return val.getType().cast<ShapedType>().getRank() == 1;
  }

  bool isScalarOp(IdentityOp identityOp) const {
    return (isScalar(identityOp.getInputs()) &&
            isScalar(identityOp.getOutput()));
  }

  LogicalResult matchAndRewrite(IdentityOp identityOp,
                                PatternRewriter &rewriter) const override {
    Location loc = identityOp.getLoc();
    // Handle scalar.
    if (isScalarOp(identityOp)) {
      rewriter.replaceAllUsesWith(identityOp.getOutput(),
                                  identityOp.getInputs());
      rewriter.eraseOp(identityOp);
      return success();
    }
    // Handle memref.
    SmallVector<Value> ubs;
    size_t rank = identityOp.getOutput().getType().cast<MemRefType>().getRank();
    ArrayRef<int64_t> shapeOutput =
        identityOp.getOutput().getType().cast<MemRefType>().getShape();
    for (size_t idx = 0; idx < rank; idx++) {
      Value dim =
          rewriter.create<arith::ConstantIndexOp>(loc, shapeOutput[idx]);
      ubs.push_back(dim);
    }
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> lbs(rank, zero);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<Value> steps(rank, one);
    (void)scf::buildLoopNest(
        rewriter, loc, lbs, ubs, steps,
        [&](OpBuilder &b, Location loc, ValueRange localIvs) {
          Value input = identityOp.getInputs();
          // input is scalar.
          if (isScalar(input))
            b.create<memref::StoreOp>(loc, input, identityOp.getOutput(),
                                      localIvs);
          // input is a 1d-memref.
          else if (is1DMemRef(input)) {
            Value scalarVal = b.create<memref::LoadOp>(loc, input, localIvs[1]);
            b.create<memref::StoreOp>(loc, scalarVal, identityOp.getOutput(),
                                      localIvs);
          }
          // input is a 2d-memref.
          else {
            ArrayRef<int64_t> shapeInput =
                identityOp.getInputs().getType().cast<MemRefType>().getShape();
            SmallVector<Value, 2> inputIvs = localIvs;
            // broadcasting dimension with size 1.
            for (size_t idx = 0; idx < shapeInput.size(); idx++)
              if (shapeInput[idx] == 1)
                inputIvs[idx] = zero;
            Value scalarVal = b.create<memref::LoadOp>(loc, input, inputIvs);
            b.create<memref::StoreOp>(loc, scalarVal, identityOp.getOutput(),
                                      localIvs);
          }
        });
    rewriter.eraseOp(identityOp);
    return success();
  }
};

// Convert tpp.relu to SCF loops.
struct ConvertTppReluOp : public OpRewritePattern<ReluOp> {
  using OpRewritePattern<ReluOp>::OpRewritePattern;

  bool isScalarOp(ReluOp reluOp) const {
    return !reluOp.getOutput().getType().isa<ShapedType>();
  }

  LogicalResult matchAndRewrite(ReluOp reluOp,
                                PatternRewriter &rewriter) const override {
    Location loc = reluOp.getLoc();
    // handle scalar case.
    if (isScalarOp(reluOp)) {
      rewriter.create<arith::MaxFOp>(loc, reluOp.getOutput());
      rewriter.eraseOp(reluOp);
      return success();
    }
    // handle memref case.
    SmallVector<Value> ubs;
    size_t rank = reluOp.getOutput().getType().cast<MemRefType>().getRank();
    for (size_t idx = 0; idx < rank; idx++) {
      Value dim = rewriter.create<arith::ConstantIndexOp>(
          loc, reluOp.getOutput().getType().cast<MemRefType>().getShape()[idx]);
      ubs.push_back(dim);
    }
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> lbs(rank, zero);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<Value> steps(rank, one);

    Type elementType =
        reluOp.getOutput().getType().cast<MemRefType>().getElementType();
    Value zeroConstant = rewriter.create<arith::ConstantOp>(
        loc, elementType, rewriter.getFloatAttr(elementType, 0));

    (void)scf::buildLoopNest(
        rewriter, loc, lbs, ubs, steps,
        [&](OpBuilder &b, Location loc, ValueRange localIvs) {
          Value scalarLhs =
              b.create<memref::LoadOp>(loc, reluOp.getInputs(), localIvs);
          Value scalarRelu =
              b.create<arith::MaxFOp>(loc, zeroConstant, scalarLhs);
          b.create<memref::StoreOp>(loc, scalarRelu, reluOp.getOutput(),
                                    localIvs);
        });

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
      return rewriter.notifyMatchFailure(matmulOp, "Packed BF16 loops unsupported");
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
          Value scalarC = b.create<memref::LoadOp>(loc, matmulOp.getOutput(),
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
          Value scalarC = b.create<memref::LoadOp>(loc, brgemmOp.getOutput(),
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
