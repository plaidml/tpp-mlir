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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

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
// or
//
// arith.addf(%a, %b)
//
struct ConvertTppAddOp : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  bool isScalarOp(AddOp addOp) const {
    return !addOp.getLhs().getType().isa<ShapedType>();
  }

  LogicalResult matchAndRewrite(AddOp addOp,
                                PatternRewriter &rewriter) const override {
    Location loc = addOp.getLoc();
    // handle scalar case.
    if (isScalarOp(addOp)) {
      Value scalarAdd =
          rewriter.create<arith::AddFOp>(loc, addOp.getLhs(), addOp.getRhs());
      addOp.getOutput().replaceAllUsesWith(scalarAdd);
      rewriter.eraseOp(addOp);
      return success();
    }
    // handle memref case.
    SmallVector<Value> ubs;
    size_t rank = addOp.getLhs().getType().cast<MemRefType>().getRank();
    for (size_t idx = 0; idx < rank; idx++) {
      Value dim = rewriter.create<arith::ConstantIndexOp>(
          loc, addOp.getLhs().getType().cast<MemRefType>().getShape()[idx]);
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
              b.create<memref::LoadOp>(loc, addOp.getLhs(), localIvs);
          Value scalarRhs =
              b.create<memref::LoadOp>(loc, addOp.getRhs(), localIvs);
          Value addLhsAndRhs =
              b.create<arith::AddFOp>(loc, scalarLhs, scalarRhs);
          b.create<memref::StoreOp>(loc, addLhsAndRhs, addOp.getOutput(),
                                    localIvs);
        });
    rewriter.eraseOp(addOp);
    return success();
  }
};

// Converts identity op.
struct ConvertTppIdentityOp : public OpRewritePattern<IdentityOp> {
  using OpRewritePattern<IdentityOp>::OpRewritePattern;

  bool isScalar(Value val) const { return !val.getType().isa<ShapedType>(); }

  bool is1DMemRef(Value val) const {
    if (isScalar(val))
      return false;
    return val.getType().cast<ShapedType>().getRank() == 1;
  }

  bool isScalarOp(IdentityOp identityOp) const {
    return (isScalar(identityOp.getInput()) &&
            isScalar(identityOp.getOutput()));
  }

  LogicalResult matchAndRewrite(IdentityOp identityOp,
                                PatternRewriter &rewriter) const override {
    Location loc = identityOp.getLoc();
    // Handle scalar.
    if (isScalarOp(identityOp)) {
      identityOp.getOutput().replaceAllUsesWith(identityOp.getInput());
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
          Value input = identityOp.getInput();
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
                identityOp.getInput().getType().cast<MemRefType>().getShape();
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

// Convert relu to loops.
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
              b.create<memref::LoadOp>(loc, reluOp.getOutput(), localIvs);
          Value scalarRelu =
              b.create<arith::MaxFOp>(loc, zeroConstant, scalarLhs);
          b.create<memref::StoreOp>(loc, scalarRelu, reluOp.getOutput(),
                                    localIvs);
        });

    rewriter.eraseOp(reluOp);
    return success();
  }
};

// Convert matmul to loops.
struct ConvertTppMatmulOp : public OpRewritePattern<MatmulOp> {
  using OpRewritePattern<MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    Location loc = matmulOp.getLoc();
    ArrayRef<int64_t> shapeC = matmulOp.getMatrixCType().getShape();
    ArrayRef<int64_t> shapeA =
        matmulOp.getMatrixA().getType().cast<MemRefType>().getShape();
    if (shapeA.size() == 3)
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
          Value scalarA = b.create<memref::LoadOp>(loc, matmulOp.getMatrixA(),
                                                   ValueRange{localI, localK});
          Value scalarB = b.create<memref::LoadOp>(loc, matmulOp.getMatrixB(),
                                                   ValueRange{localK, localJ});
          Value scalarC = b.create<memref::LoadOp>(loc, matmulOp.getMatrixC(),
                                                   ValueRange{localI, localJ});
          Value scalarMul = b.create<arith::MulFOp>(loc, scalarA, scalarB);
          Value scalarAdd = b.create<arith::AddFOp>(loc, scalarC, scalarMul);
          b.create<memref::StoreOp>(loc, scalarAdd, matmulOp.getMatrixC(),
                                    ValueRange{localI, localJ});
        });

    rewriter.eraseOp(matmulOp);
    return success();
  }
};

// Convert brgemm to loops.
struct ConvertTppBrgemmOp : public OpRewritePattern<BrgemmOp> {
  using OpRewritePattern<BrgemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BrgemmOp brgemmOp,
                                PatternRewriter &rewriter) const override {
    Location loc = brgemmOp.getLoc();
    ArrayRef<int64_t> shapeC = brgemmOp.getMatrixCType().getShape();
    ArrayRef<int64_t> shapeA = brgemmOp.getBatchMatrixAType().getShape();
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
          Value scalarA =
              b.create<memref::LoadOp>(loc, brgemmOp.getBatchMatrixA(),
                                       ValueRange{localB, localI, localK});
          Value scalarB =
              b.create<memref::LoadOp>(loc, brgemmOp.getBatchMatrixB(),
                                       ValueRange{localB, localK, localJ});
          Value scalarC = b.create<memref::LoadOp>(loc, brgemmOp.getMatrixC(),
                                                   ValueRange{localI, localJ});
          Value scalarMul = b.create<arith::MulFOp>(loc, scalarA, scalarB);
          Value scalarAdd = b.create<arith::AddFOp>(loc, scalarC, scalarMul);
          b.create<memref::StoreOp>(loc, scalarAdd, brgemmOp.getMatrixC(),
                                    ValueRange{localI, localJ});
        });
    rewriter.eraseOp(brgemmOp);
    return success();
  }
};

// Convert offset brgemm to loops.
struct ConvertTppOffsetBrgemmOp : public OpRewritePattern<OffsetBrgemmOp> {
  using OpRewritePattern<OffsetBrgemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(OffsetBrgemmOp brgemmOp,
                                PatternRewriter &rewriter) const override {
    ArrayRef<int64_t> offsetsInA = brgemmOp.getOffsetsA();
    ArrayRef<int64_t> offsetsInB = brgemmOp.getOffsetsB();
    assert(offsetsInA.size() != offsetsInB.size() &&
           "expect same size for offsets");

    ArrayRef<int64_t> leadingDims = brgemmOp.getLdims();
    ArrayRef<int64_t> dims = brgemmOp.getDims();

    Location loc = brgemmOp.getLoc();
    Value basePtrA = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, brgemmOp.getMemrefA());
    Value basePtrB = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, brgemmOp.getMemrefB());
    Value basePtrC = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        loc, brgemmOp.getMemrefC());
    for (int idxInOffsets = 0, sizeOffsets = offsetsInA.size();
         idxInOffsets < sizeOffsets; idxInOffsets++) {
      
      // Create constant from offsets.
      Value ctsOffsetInA = rewriter.create<arith::ConstantIndexOp>(
          loc, offsetsInA[idxInOffsets]); 
      Value ctsOffsetInB = rewriter.create<arith::ConstantIndexOp>(
          loc, offsetsInB[idxInOffsets]);

      // Add constants to base pointers.
      basePtrA = rewriter.create<arith::AddIOp>(loc, basePtrA, ctsOffsetInA);
      basePtrB = rewriter.create<arith::AddIOp>(loc, basePtrB, ctsOffsetInB);

      // Cast Index tp i64.
      Value castedOffsetInA = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), ctsOffsetInA);
      Value castedOffsetInB = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), ctsOffsetInB);
      
      // Create llvm.inttoptr using memref element type.
      Type basePtrType =
          LLVM::LLVMPointerType::get(brgemmOp.getMemrefA().getType().getElementType());
      Value baseLLVMPtrA = rewriter.create<LLVM::IntToPtrOp>(loc, basePtrType, castedOffsetInA);
      Value baseLLVMPtrB = rewriter.create<LLVM::IntToPtrOp>(loc, basePtrType, castedOffsetInB);

      // Call a naive matmul.
      
    
    }
    return failure();
  }
};

void populateTppToLoopsPatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<ConvertTppAddOp, 
               ConvertTppIdentityOp,
               ConvertTppMatmulOp,
               ConvertTppBrgemmOp,
               ConvertTppOffsetBrgemmOp,
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
