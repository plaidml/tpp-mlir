//===- ConvertTppToXsmm.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Tpp/TppOps.h"
#include "TPP/Dialect/Xsmm/XsmmAttr.h"
#include "TPP/Dialect/Xsmm/XsmmOps.h"
#include "TPP/Passes.h"
#include "TPP/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

static FailureOr<int64_t> getLeadingDim(MemRefType memref, size_t pos = 0) {
  // For 1d memref we cannot use the stride as leading dimension, but the
  // leading dimension is the dimension itself.
  if (memref.getRank() == 1)
    return memref.getShape()[0];

  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(getStridesAndOffset(memref, strides, offset)))
    return failure();
  // fail if the strides are non-constant
  if (llvm::any_of(strides, [](int64_t stride) {
        return stride == ShapedType::kDynamic;
      }))
    return failure();
  return strides[pos];
}

struct ConvertTppMatmulOp : public OpRewritePattern<tpp::MatmulOp> {
  using OpRewritePattern<tpp::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpp::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    Location loc = matmulOp.getLoc();

    MemRefType memrefC = matmulOp.getMatrixCType();
    MemRefType memrefA = matmulOp.getMatrixAType();
    MemRefType memrefB = matmulOp.getMatrixBType();
    int64_t m = memrefC.getShape()[0];
    int64_t n = memrefC.getShape()[1];
    int64_t k = memrefA.getShape()[1];
    auto ldaDim = getLeadingDim(memrefA);
    if (failed(ldaDim))
      return rewriter.notifyMatchFailure(matmulOp, "Cannot compute lda");
    int64_t lda = *ldaDim;

    auto ldbDim = getLeadingDim(memrefB);
    if (failed(ldbDim))
      return rewriter.notifyMatchFailure(matmulOp, "Cannot compute ldb");
    int64_t ldb = *ldbDim;

    auto ldcDim = getLeadingDim(memrefC);
    if (failed(ldcDim))
      return rewriter.notifyMatchFailure(matmulOp, "Cannot compute ldc");
    int64_t ldc = *ldcDim;

    IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
    DenseI64ArrayAttr dims = DenseI64ArrayAttr::get(
        rewriter.getContext(), ArrayRef<int64_t>{m, n, k, lda, ldb, ldc});
    xsmm::TernaryKindAttr attr = xsmm::TernaryKindAttr::get(
        matmulOp.getContext(), xsmm::TernaryKind::MATMUL);
    xsmm::DataTypeAttr dtype;
    if (memrefC.getElementType().isBF16()) {
      dtype =
          xsmm::DataTypeAttr::get(matmulOp.getContext(), xsmm::DataType::BF16);
    } else {
      assert(memrefC.getElementType().isF32() &&
             "Element type neither bf16 nor f32");
      dtype =
          xsmm::DataTypeAttr::get(matmulOp.getContext(), xsmm::DataType::F32);
    }

    Value dispatched = rewriter.create<xsmm::TernaryDispatchOp>(
        loc, integer64, attr, dims, dtype,
        BoolAttr::get(matmulOp.getContext(), false));

    SmallVector<Value, 6> invokeOperands;
    invokeOperands.push_back(dispatched);
    invokeOperands.append(matmulOp->getOperands().begin(),
                          matmulOp->getOperands().end());
    rewriter.replaceOpWithNewOp<xsmm::TernaryOp>(matmulOp, dtype, attr,
                                                 invokeOperands);
    return success();
  }
};

struct ConvertTppVNNIMatmulOp : public OpRewritePattern<tpp::VNNIMatmulOp> {
  using OpRewritePattern<tpp::VNNIMatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpp::VNNIMatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    Location loc = matmulOp.getLoc();

    MemRefType memrefC = matmulOp.getMatrixCType();
    MemRefType memrefA = matmulOp.getMatrixAType();
    MemRefType memrefB = matmulOp.getMatrixBType();
    int64_t m = memrefC.getShape()[0];
    int64_t n = memrefC.getShape()[1];
    int64_t k = memrefA.getShape()[1];
    auto ldaDim = getLeadingDim(memrefA);
    if (failed(ldaDim))
      return rewriter.notifyMatchFailure(matmulOp, "Cannot compute lda");
    int64_t lda = *ldaDim;

    auto ldbDim = getLeadingDim(memrefB);
    if (failed(ldbDim))
      return rewriter.notifyMatchFailure(matmulOp, "Cannot compute ldb");
    int64_t ldb = *ldbDim;
    auto divLdbDim = getLeadingDim(memrefB, 1);
    if (failed(divLdbDim))
      return rewriter.notifyMatchFailure(matmulOp, "Cannot compute ldb");
    ldb = ldb / (*divLdbDim);

    auto ldcDim = getLeadingDim(memrefC);
    if (failed(ldcDim))
      return rewriter.notifyMatchFailure(matmulOp, "Cannot compute ldc");
    int64_t ldc = *ldcDim;

    IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
    DenseI64ArrayAttr dims = DenseI64ArrayAttr::get(
        rewriter.getContext(), ArrayRef<int64_t>{m, n, k, lda, ldb, ldc});
    xsmm::TernaryKindAttr attr = xsmm::TernaryKindAttr::get(
        matmulOp.getContext(), xsmm::TernaryKind::MATMUL);
    xsmm::DataTypeAttr dtype =
        xsmm::DataTypeAttr::get(matmulOp.getContext(), xsmm::DataType::BF16);
    Value dispatched = rewriter.create<xsmm::TernaryDispatchOp>(
        loc, integer64, attr, dims, dtype,
        BoolAttr::get(matmulOp.getContext(), true));

    SmallVector<Value, 6> invokeOperands;
    invokeOperands.push_back(dispatched);
    invokeOperands.append(matmulOp->getOperands().begin(),
                          matmulOp->getOperands().end());
    rewriter.replaceOpWithNewOp<xsmm::TernaryOp>(matmulOp, dtype, attr,
                                                 invokeOperands);
    return success();
  }
};

struct ConvertTppBrgemmOp : public OpRewritePattern<tpp::BrgemmOp> {
  using OpRewritePattern<tpp::BrgemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpp::BrgemmOp brgemmOp,
                                PatternRewriter &rewriter) const override {
    Location loc = brgemmOp.getLoc();

    MemRefType memrefC = brgemmOp.getMatrixCType();
    MemRefType memrefA = brgemmOp.getBatchMatrixAType();
    MemRefType memrefB = brgemmOp.getBatchMatrixBType();
    int64_t m = memrefC.getShape()[0];
    int64_t n = memrefC.getShape()[1];
    int64_t k = memrefA.getShape()[2];
    int64_t batchSize = memrefB.getShape()[0];

    auto ldaDim = getLeadingDim(memrefA, 1);
    if (failed(ldaDim))
      return rewriter.notifyMatchFailure(brgemmOp, "Cannot compute lda");
    int64_t lda = *ldaDim;
    auto ldbDim = getLeadingDim(memrefB, 1);
    if (failed(ldbDim))
      return rewriter.notifyMatchFailure(brgemmOp, "Cannot compute ldb");
    int64_t ldb = *ldbDim;

    auto ldcDim = getLeadingDim(memrefC);
    if (failed(ldcDim))
      return rewriter.notifyMatchFailure(brgemmOp, "Cannot compute ldc");
    int64_t ldc = *ldcDim;

    IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
    DenseI64ArrayAttr dims = DenseI64ArrayAttr::get(
        rewriter.getContext(), ArrayRef<int64_t>{m, n, k, lda, ldb, ldc});
    xsmm::TernaryKindAttr attr = xsmm::TernaryKindAttr::get(
        brgemmOp.getContext(), xsmm::TernaryKind::BRGEMM);
    xsmm::DataTypeAttr dtype =
        xsmm::DataTypeAttr::get(brgemmOp.getContext(), xsmm::DataType::F32);
    if (memrefC.getElementType().isBF16()) {
      dtype =
          xsmm::DataTypeAttr::get(brgemmOp.getContext(), xsmm::DataType::BF16);
    } else {
      assert(memrefC.getElementType().isF32() &&
             "Element type neither bf16 nor f32");
      dtype =
          xsmm::DataTypeAttr::get(brgemmOp.getContext(), xsmm::DataType::F32);
    }

    Value dispatched = rewriter.create<xsmm::TernaryDispatchOp>(
        loc, integer64, attr, dims, dtype,
        BoolAttr::get(brgemmOp.getContext(), false));
    Value batchDim = rewriter.create<arith::ConstantOp>(
        loc, integer64, rewriter.getIntegerAttr(integer64, batchSize));
    SmallVector<Value, 6> invokeOperands;
    invokeOperands.push_back(dispatched);
    invokeOperands.append(brgemmOp->getOperands().begin(),
                          brgemmOp->getOperands().end());
    invokeOperands.push_back(batchDim);
    rewriter.replaceOpWithNewOp<xsmm::TernaryOp>(brgemmOp, dtype, attr,
                                                 invokeOperands);
    return success();
  }
};

struct ConvertTppVNNIBrgemmOp : public OpRewritePattern<tpp::VNNIBrgemmOp> {
  using OpRewritePattern<tpp::VNNIBrgemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpp::VNNIBrgemmOp brgemmOp,
                                PatternRewriter &rewriter) const override {
    Location loc = brgemmOp.getLoc();

    MemRefType memrefC = brgemmOp.getMatrixCType();
    MemRefType memrefA = brgemmOp.getBatchMatrixAType();
    MemRefType memrefB = brgemmOp.getBatchMatrixBType();
    int64_t m = memrefC.getShape()[0];
    int64_t n = memrefC.getShape()[1];
    int64_t k = memrefA.getShape()[2];
    int64_t batchSize = memrefB.getShape()[0];

    auto ldaDim = getLeadingDim(memrefA, 1);
    if (failed(ldaDim))
      return rewriter.notifyMatchFailure(brgemmOp, "Cannot compute lda");
    int64_t lda = *ldaDim;

    auto ldbDim = getLeadingDim(memrefB, 1);
    if (failed(ldbDim))
      return rewriter.notifyMatchFailure(brgemmOp, "Cannot compute ldb");
    int64_t ldb = *ldbDim;
    auto divLdbDim = getLeadingDim(memrefB, 2);
    if (failed(divLdbDim))
      return rewriter.notifyMatchFailure(brgemmOp, "Cannot compute ldb");
    ldb = ldb / (*divLdbDim);

    auto ldcDim = getLeadingDim(memrefC);
    if (failed(ldcDim))
      return rewriter.notifyMatchFailure(brgemmOp, "Cannot compute ldc");
    int64_t ldc = *ldcDim;

    IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
    DenseI64ArrayAttr dims = DenseI64ArrayAttr::get(
        rewriter.getContext(), ArrayRef<int64_t>{m, n, k, lda, ldb, ldc});
    xsmm::TernaryKindAttr attr = xsmm::TernaryKindAttr::get(
        brgemmOp.getContext(), xsmm::TernaryKind::BRGEMM);
    xsmm::DataTypeAttr dtype =
        xsmm::DataTypeAttr::get(brgemmOp.getContext(), xsmm::DataType::BF16);

    Value dispatched = rewriter.create<xsmm::TernaryDispatchOp>(
        loc, integer64, attr, dims, dtype,
        BoolAttr::get(brgemmOp.getContext(), true));
    Value batchDim = rewriter.create<arith::ConstantOp>(
        loc, integer64, rewriter.getIntegerAttr(integer64, batchSize));
    SmallVector<Value, 6> invokeOperands;
    invokeOperands.push_back(dispatched);
    invokeOperands.append(brgemmOp->getOperands().begin(),
                          brgemmOp->getOperands().end());
    invokeOperands.push_back(batchDim);
    rewriter.replaceOpWithNewOp<xsmm::TernaryOp>(brgemmOp, dtype, attr,
                                                 invokeOperands);
    return success();
  }
};

struct ConvertTppFusedBrgemmOp : public OpRewritePattern<tpp::FusedBrgemmOp> {
  using OpRewritePattern<tpp::FusedBrgemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpp::FusedBrgemmOp brgemmOp,
                                PatternRewriter &rewriter) const override {
    Location loc = brgemmOp.getLoc();

    MemRefType memrefC = brgemmOp.getMatrixCType();
    MemRefType memrefA = brgemmOp.getBatchMatrixAType();
    MemRefType memrefB = brgemmOp.getBatchMatrixBType();
    int64_t m = memrefC.getShape()[0];
    int64_t n = memrefC.getShape()[1];
    int64_t k = memrefA.getShape()[2];
    int64_t batchSize = memrefB.getShape()[0];

    auto ldaDim = getLeadingDim(memrefA, 1);
    if (failed(ldaDim))
      return rewriter.notifyMatchFailure(brgemmOp, "Cannot compute lda");
    int64_t lda = *ldaDim;

    auto ldbDim = getLeadingDim(memrefB, 1);
    if (failed(ldbDim))
      return rewriter.notifyMatchFailure(brgemmOp, "Cannot compute ldb");
    int64_t ldb = *ldbDim;
    auto divLdbDim = getLeadingDim(memrefB, 2);
    if (failed(divLdbDim))
      return rewriter.notifyMatchFailure(brgemmOp, "Cannot compute ldb");
    ldb = ldb / (*divLdbDim);

    auto ldcDim = getLeadingDim(memrefC);
    if (failed(ldcDim))
      return rewriter.notifyMatchFailure(brgemmOp, "Cannot compute ldc");
    int64_t ldc = *ldcDim;

    IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
    DenseI64ArrayAttr dims = DenseI64ArrayAttr::get(
        rewriter.getContext(), ArrayRef<int64_t>{m, n, k, lda, ldb, ldc});
    xsmm::QuarternaryKindAttr attr = xsmm::QuarternaryKindAttr::get(
        brgemmOp.getContext(), xsmm::QuarternaryKind::FUSED_BRGEMM);
    xsmm::DataTypeAttr dtype;
    if (memrefC.cast<ShapedType>().getElementType().isBF16()) {
      dtype =
          xsmm::DataTypeAttr::get(brgemmOp.getContext(), xsmm::DataType::BF16);
    } else {
      assert(memrefC.cast<ShapedType>().getElementType().isF32() &&
             "Element type neither bf16 nor f32");
      dtype =
          xsmm::DataTypeAttr::get(brgemmOp.getContext(), xsmm::DataType::F32);
    }

    Value dispatched = rewriter.create<xsmm::QuarternaryDispatchOp>(
        loc, integer64, attr, dims, dtype,
        BoolAttr::get(brgemmOp.getContext(), false));
    Value batchDim = rewriter.create<arith::ConstantOp>(
        loc, integer64, rewriter.getIntegerAttr(integer64, batchSize));
    SmallVector<Value, 7> invokeOperands;
    invokeOperands.push_back(dispatched);
    invokeOperands.append(brgemmOp->getOperands().begin(),
                          brgemmOp->getOperands().end());
    invokeOperands.push_back(batchDim);
    rewriter.replaceOpWithNewOp<xsmm::QuarternaryOp>(brgemmOp, dtype, attr,
                                                     invokeOperands);
    return success();
  }
};

struct ConvertTppFusedVNNIBrgemmOp
    : public OpRewritePattern<tpp::FusedVNNIBrgemmOp> {
  using OpRewritePattern<tpp::FusedVNNIBrgemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpp::FusedVNNIBrgemmOp brgemmOp,
                                PatternRewriter &rewriter) const override {
    Location loc = brgemmOp.getLoc();

    MemRefType memrefC = brgemmOp.getMatrixCType();
    MemRefType memrefA = brgemmOp.getBatchMatrixAType();
    MemRefType memrefB = brgemmOp.getBatchMatrixBType();
    int64_t m = memrefC.getShape()[0];
    int64_t n = memrefC.getShape()[1];
    int64_t k = memrefA.getShape()[2];
    int64_t batchSize = memrefB.getShape()[0];

    auto ldaDim = getLeadingDim(memrefA, 1);
    if (failed(ldaDim))
      return rewriter.notifyMatchFailure(brgemmOp, "Cannot compute lda");
    int64_t lda = *ldaDim;

    auto ldbDim = getLeadingDim(memrefB, 1);
    if (failed(ldbDim))
      return rewriter.notifyMatchFailure(brgemmOp, "Cannot compute ldb");
    int64_t ldb = *ldbDim;
    auto divLdbDim = getLeadingDim(memrefB, 2);
    if (failed(divLdbDim))
      return rewriter.notifyMatchFailure(brgemmOp, "Cannot compute ldb");
    ldb = ldb / (*divLdbDim);

    auto ldcDim = getLeadingDim(memrefC);
    if (failed(ldcDim))
      return rewriter.notifyMatchFailure(brgemmOp, "Cannot compute ldc");
    int64_t ldc = *ldcDim;

    IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
    DenseI64ArrayAttr dims = DenseI64ArrayAttr::get(
        rewriter.getContext(), ArrayRef<int64_t>{m, n, k, lda, ldb, ldc});
    xsmm::QuarternaryKindAttr attr = xsmm::QuarternaryKindAttr::get(
        brgemmOp.getContext(), xsmm::QuarternaryKind::FUSED_BRGEMM);
    xsmm::DataTypeAttr dtype =
        xsmm::DataTypeAttr::get(brgemmOp.getContext(), xsmm::DataType::BF16);

    Value dispatched = rewriter.create<xsmm::QuarternaryDispatchOp>(
        loc, integer64, attr, dims, dtype,
        BoolAttr::get(brgemmOp.getContext(), true));
    Value batchDim = rewriter.create<arith::ConstantOp>(
        loc, integer64, rewriter.getIntegerAttr(integer64, batchSize));
    SmallVector<Value, 7> invokeOperands;
    invokeOperands.push_back(dispatched);
    invokeOperands.append(brgemmOp->getOperands().begin(),
                          brgemmOp->getOperands().end());
    invokeOperands.push_back(batchDim);
    rewriter.replaceOpWithNewOp<xsmm::QuarternaryOp>(brgemmOp, dtype, attr,
                                                     invokeOperands);
    return success();
  }
};

struct ConvertTppIdentityOp : public OpRewritePattern<tpp::IdentityOp> {
  using OpRewritePattern<tpp::IdentityOp>::OpRewritePattern;

  // Examples:
  // If lower=[c], higher=[a, b, c], [c] reshaped into [1, 1, c].
  // If lower=[b, c], higher=[a, b, c], [b, c] reshaped into [1, b, c].
  // If lower=[a], higher=[a, a], [a] reshaped into [1, a].
  // If lower=[a], target=[a, b, a], [a] reshaped into [1, 1, a].
  // If lower=[], target=[a, b, c], [] reshaped into [1, 1, 1].
  void
  computeBcastShapeInput(ArrayRef<int64_t> higherRankShape,
                         ArrayRef<int64_t> lowerRankShape,
                         SmallVectorImpl<int64_t> &reshapeOutputShape) const {
    // Initialize new shapes with [1] * higherRank.
    int64_t higherRank = higherRankShape.size();
    int64_t lowerRank = lowerRankShape.size();

    reshapeOutputShape.assign(higherRank, 1);

    int64_t higherRankDim;
    int64_t lowerRankDim;

    for (int64_t i = higherRank - 1, j = lowerRank - 1; i >= 0 && j >= 0;
         i--, j--) {
      higherRankDim = higherRankShape[i];
      lowerRankDim = lowerRankShape[j];

      if (lowerRankDim == 1 && higherRankDim > 1)
        reshapeOutputShape[i] = 1;
      else if ((lowerRankDim > 1 && higherRankDim == 1) ||
               (lowerRankDim == higherRankDim))
        reshapeOutputShape[i] = lowerRankDim;
      else if (higherRankDim != lowerRankDim)
        assert(false && "bCast semantics for identity op broken");
    }
  }

  // Return ldi and bCast.
  std::pair<int64_t, xsmm::UnaryFlags>
  getLdiAndBCast(tpp::IdentityOp identityOp, int64_t ldo) const {
    Type inputType = identityOp.getInput().getType();

    // There are multiple ways to define a scalar.  f32, memref<1x1xf32> or
    // memref<f32>. Handle f32, and memref<1x1xf32>. memref<f32> is not allowed
    // in tpp at the moment.
    if (!inputType.isa<ShapedType>()) {
      xsmm::UnaryFlags bCast = xsmm::UnaryFlags::BCAST_SCALAR;
      int64_t ldi = 1;
      return {ldi, bCast};
    }
    ArrayRef<int64_t> shapeInput = inputType.cast<ShapedType>().getShape();
    auto isOne = [](int64_t val) { return val == 1; };
    if (llvm::all_of(shapeInput, isOne)) {
      xsmm::UnaryFlags bCast = xsmm::UnaryFlags::BCAST_SCALAR;
      int64_t ldi = 1;
      return {ldi, bCast};
    }

    Type outputType = identityOp.getOutput().getType();

    ArrayRef<int64_t> shapeOutput = outputType.cast<ShapedType>().getShape();
    assert(shapeOutput.size() >= shapeInput.size() &&
           "output rank must be >= input rank");
    SmallVector<int64_t, 4> bShapeInput;
    computeBcastShapeInput(shapeOutput, shapeInput, bShapeInput);
    assert(shapeOutput.size() == bShapeInput.size());
    shapeInput = bShapeInput;

    if (shapeInput[1] == 1 && shapeOutput[1] > 1) {
      xsmm::UnaryFlags bCast = xsmm::UnaryFlags::BCAST_ROW;
      int64_t ldi = *getLeadingDim(outputType.cast<MemRefType>(), 1);
      return {ldi, bCast};
    }

    if (shapeInput[0] == 1 && shapeOutput[0] > 1) {
      xsmm::UnaryFlags bCast = xsmm::UnaryFlags::BCAST_COL;
      int64_t ldi = *getLeadingDim(outputType.cast<MemRefType>());
      return {ldi, bCast};
    }

    if (shapeInput[0] == shapeOutput[0] && shapeInput[1] == shapeOutput[1]) {
      xsmm::UnaryFlags bCast = xsmm::UnaryFlags::NONE;
      int64_t ldi = *getLeadingDim(inputType.cast<MemRefType>());
      return {ldi, bCast};
    }
    assert(false && "failed to get ldi and bCast for identity");
  }

  LogicalResult matchAndRewrite(tpp::IdentityOp identityOp,
                                PatternRewriter &rewriter) const override {
    Location loc = identityOp.getLoc();
    // no conversion if identity is a scalar operation.
    Type outputType = identityOp.getOutput().getType();
    MemRefType outputMemRefType = outputType.dyn_cast<MemRefType>();
    if (!outputMemRefType || outputMemRefType.getRank() != 2)
      return rewriter.notifyMatchFailure(identityOp, "not a 2-D memref type");

    int64_t outputOffset;
    SmallVector<int64_t> outputStrides;
    if (failed(
            getStridesAndOffset(outputMemRefType, outputStrides, outputOffset)))
      return rewriter.notifyMatchFailure(identityOp, "not a strided memref");
    if (outputStrides.back() != 1)
      return rewriter.notifyMatchFailure(identityOp,
                                         "most minor stride is != 1");

    int64_t m = outputMemRefType.getShape()[0];
    int64_t n = outputMemRefType.getShape()[1];
    int64_t ldo = outputStrides.front();
    std::pair<int64_t, xsmm::UnaryFlags> ldiAndBCast =
        getLdiAndBCast(identityOp, ldo);
    int64_t ldi = ldiAndBCast.first;
    xsmm::UnaryFlags bCast = ldiAndBCast.second;
    IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
    xsmm::UnaryKindAttr attr = xsmm::UnaryKindAttr::get(
        identityOp.getContext(), xsmm::UnaryKind::IDENTITY);
    DenseI64ArrayAttr dims = DenseI64ArrayAttr::get(
        rewriter.getContext(), ArrayRef<int64_t>{m, n, ldi, ldo});
    xsmm::UnaryFlagsAttr bCastAttr =
        xsmm::UnaryFlagsAttr::get(identityOp.getContext(), bCast);
    xsmm::DataTypeAttr dtype;
    if (outputMemRefType.getElementType().isBF16()) {
      dtype = xsmm::DataTypeAttr::get(identityOp.getContext(),
                                      xsmm::DataType::BF16);
    } else {
      assert(outputMemRefType.getElementType().isF32() &&
             "Element type neither bf16 nor f32");
      dtype =
          xsmm::DataTypeAttr::get(identityOp.getContext(), xsmm::DataType::F32);
    }

    Value dispatched = rewriter.create<xsmm::UnaryDispatchOp>(
        loc, integer64, attr, dims, bCastAttr, dtype);

    SmallVector<Value, 6> invokeOperands;
    invokeOperands.push_back(dispatched);
    invokeOperands.append(identityOp->getOperands().begin(),
                          identityOp->getOperands().end());

    rewriter.replaceOpWithNewOp<xsmm::UnaryOp>(identityOp, dtype, attr,
                                               invokeOperands);
    return success();
  }
};

struct ConvertTppReluOp : public OpRewritePattern<tpp::ReluOp> {
  using OpRewritePattern<tpp::ReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpp::ReluOp reluOp,
                                PatternRewriter &rewriter) const override {
    Location loc = reluOp.getLoc();
    Type outputType = reluOp.getInput().getType();
    assert(outputType.isa<MemRefType>() && "expect a memref type");

    MemRefType outputMemRef = outputType.cast<MemRefType>();
    assert((outputMemRef.getRank() == 1 || outputMemRef.getRank() == 2) &&
           "expect memref with rank 1 or 2");

    int64_t m = (outputMemRef.getRank() == 2) ? outputMemRef.getShape()[0] : 1;
    int64_t n = (outputMemRef.getRank() == 2) ? outputMemRef.getShape()[1]
                                              : outputMemRef.getShape()[0];

    auto leadDim = getLeadingDim(outputMemRef);
    if (failed(leadDim))
      return rewriter.notifyMatchFailure(reluOp, "Cannot compute ldo/ldi");
    int64_t ldo = *leadDim;
    int64_t ldi = *leadDim;

    xsmm::UnaryFlags bCast = xsmm::UnaryFlags::NONE;
    xsmm::UnaryKindAttr attr =
        xsmm::UnaryKindAttr::get(reluOp.getContext(), xsmm::UnaryKind::RELU);
    DenseI64ArrayAttr dims = DenseI64ArrayAttr::get(
        rewriter.getContext(), ArrayRef<int64_t>{m, n, ldi, ldo});
    xsmm::UnaryFlagsAttr bCastAttr =
        xsmm::UnaryFlagsAttr::get(reluOp.getContext(), bCast);
    IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
    xsmm::DataTypeAttr dtype;
    if (outputMemRef.getElementType().isBF16()) {
      dtype =
          xsmm::DataTypeAttr::get(reluOp.getContext(), xsmm::DataType::BF16);
    } else {
      assert(outputMemRef.getElementType().isF32() &&
             "Element type neither bf16 nor f32");
      dtype = xsmm::DataTypeAttr::get(reluOp.getContext(), xsmm::DataType::F32);
    }

    Value dispatched = rewriter.create<xsmm::UnaryDispatchOp>(
        loc, integer64, attr, dims, bCastAttr, dtype);

    SmallVector<Value, 6> invokeOperands;
    invokeOperands.push_back(dispatched);
    invokeOperands.append(reluOp->getOperands().begin(),
                          reluOp->getOperands().end());

    rewriter.replaceOpWithNewOp<xsmm::UnaryOp>(reluOp, dtype, attr,
                                               invokeOperands);
    return success();
  }
};

struct ConvertTppAddOp : public OpRewritePattern<tpp::AddOp> {
  using OpRewritePattern<tpp::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpp::AddOp addOp,
                                PatternRewriter &rewriter) const override {
    Location loc = addOp.getLoc();
    Type outputType = addOp.getOut().getType();
    assert(outputType.isa<MemRefType>() && "expect a memref type");
    MemRefType outputMemRef = outputType.cast<MemRefType>();
    assert((outputMemRef.getRank() == 1 || outputMemRef.getRank() == 2) &&
           "expect memref with rank 1 or 2");

    int64_t m = (outputMemRef.getRank() == 2) ? outputMemRef.getShape()[0] : 1;
    int64_t n = (outputMemRef.getRank() == 2) ? outputMemRef.getShape()[1]
                                              : outputMemRef.getShape()[0];

    auto ldiLhsDim = getLeadingDim(addOp.getLhs().getType().cast<MemRefType>());
    if (failed(ldiLhsDim))
      return rewriter.notifyMatchFailure(addOp, "Cannot compute ldi on lhs");
    int64_t ldiLhs = *ldiLhsDim;

    auto ldiRhsDim = getLeadingDim(addOp.getRhs().getType().cast<MemRefType>());
    if (failed(ldiRhsDim))
      return rewriter.notifyMatchFailure(addOp, "Cannot compute ldi on rhs");
    int64_t ldiRhs = *ldiRhsDim;

    auto ldoDim = getLeadingDim(outputMemRef);
    if (failed(ldoDim))
      return rewriter.notifyMatchFailure(addOp, "Cannot compute ldo");
    int64_t ldo = *ldoDim;

    xsmm::BinaryFlags bCast = xsmm::BinaryFlags::NONE;
    xsmm::BinaryKindAttr attr =
        xsmm::BinaryKindAttr::get(addOp.getContext(), xsmm::BinaryKind::ADD);
    DenseI64ArrayAttr dims = DenseI64ArrayAttr::get(
        rewriter.getContext(), ArrayRef<int64_t>{m, n, ldiLhs, ldiRhs, ldo});
    xsmm::BinaryFlagsAttr bCastAttr =
        xsmm::BinaryFlagsAttr::get(addOp.getContext(), bCast);
    IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
    xsmm::DataTypeAttr dtype;
    if (outputMemRef.getElementType().isBF16()) {
      dtype = xsmm::DataTypeAttr::get(addOp.getContext(), xsmm::DataType::BF16);
    } else {
      assert(outputMemRef.getElementType().isF32() &&
             "Element type neither bf16 nor f32");
      dtype = xsmm::DataTypeAttr::get(addOp.getContext(), xsmm::DataType::F32);
    }

    Value dispatched = rewriter.create<xsmm::BinaryDispatchOp>(
        loc, integer64, attr, dims, bCastAttr, dtype);

    SmallVector<Value, 6> invokeOperands;
    invokeOperands.push_back(dispatched);
    invokeOperands.append(addOp->getOperands().begin(),
                          addOp->getOperands().end());
    rewriter.replaceOpWithNewOp<xsmm::BinaryOp>(addOp, dtype, attr,
                                                invokeOperands);
    return success();
  }
};

struct ConvertTppToXsmm : public ConvertTppToXsmmBase<ConvertTppToXsmm> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    tpp::populateTppToXsmmPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // namespace

void mlir::tpp::populateTppToXsmmPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertTppIdentityOp, ConvertTppReluOp, ConvertTppAddOp,
               ConvertTppMatmulOp, ConvertTppVNNIMatmulOp, ConvertTppBrgemmOp,
               ConvertTppVNNIBrgemmOp, ConvertTppFusedVNNIBrgemmOp,
               ConvertTppFusedBrgemmOp>(patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createConvertTppToXsmmPass() {
  return std::make_unique<ConvertTppToXsmm>();
}
