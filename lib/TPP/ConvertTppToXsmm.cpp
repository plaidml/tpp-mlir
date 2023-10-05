//===- ConvertTppToXsmm.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Tpp/TppOps.h"
#include "TPP/Dialect/Tpp/TppUtils.h"
#include "TPP/Dialect/Xsmm/XsmmEnum.h"
#include "TPP/Dialect/Xsmm/XsmmOps.h"
#include "TPP/Dialect/Xsmm/XsmmUtils.h"
#include "TPP/Passes.h"
#include "TPP/Transforms.h"
#include "TPP/VNNIUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

#define DEBUG_TYPE "convert-tpp-to-xsmm"

namespace {

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

static FailureOr<int64_t> getLeadingDim(Type type, size_t pos = 0) {
  // Not shaped type, the leading dimension is the single scalar.
  if (!isa<ShapedType>(type))
    return 1;
  MemRefType memref = type.cast<MemRefType>();
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

// Examples:
// If lower=[c], higher=[a, b, c], [c] reshaped into [1, 1, c].
// If lower=[b, c], higher=[a, b, c], [b, c] reshaped into [1, b, c].
// If lower=[a], higher=[a, a], [a] reshaped into [1, a].
// If lower=[a], target=[a, b, a], [a] reshaped into [1, 1, a].
// If lower=[], target=[a, b, c], [] reshaped into [1, 1, 1].
static void
computeBcastShapeInput(ArrayRef<int64_t> higherRankShape,
                       ArrayRef<int64_t> lowerRankShape,
                       SmallVectorImpl<int64_t> &reshapeOutputShape) {
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

//===----------------------------------------------------------------------===//
// Conversions
//===----------------------------------------------------------------------===//

template <typename OpTy>
static FailureOr<DenseI64ArrayAttr>
getSizesAndLeadingDimsForGemmLikeOp(RewriterBase &rewriter, OpTy opTy) {
  assert(opTy.hasBufferSemantics() && "expects buffer semantics");

  bool isBrgemm = isa<tpp::BrgemmOp>(opTy.getOperation()) ||
                  isa<tpp::FusedBrgemmOp>(opTy.getOperation());

  auto memrefC = opTy.getOutputType();
  auto memrefA = opTy.getMemRefInputType(0);
  auto memrefB = opTy.getMemRefInputType(1);

  int64_t m = memrefC.getShape()[0];
  int64_t n = memrefC.getShape()[1];
  int64_t k = (isBrgemm) ? memrefA.getShape()[2] : memrefA.getShape()[1];

  auto ldaDim =
      (isBrgemm) ? getLeadingDim(memrefA, /*pos=*/1) : getLeadingDim(memrefA);
  if (failed(ldaDim)) {
    LLVM_DEBUG(llvm::dbgs() << "Cannot compute lda\n");
    return failure();
  }
  int64_t lda = *ldaDim;

  auto ldbDim =
      (isBrgemm) ? getLeadingDim(memrefB, /*pos=*/1) : getLeadingDim(memrefB);
  if (failed(ldbDim)) {
    LLVM_DEBUG(llvm::dbgs() << "Cannot compute ldb\n");
    return failure();
  }
  int64_t ldb = (vnni::utils::isInVnniLayout(memrefB))
                    ? *ldbDim / *vnni::utils::getVnniBlockingFactor(memrefB)
                    : *ldbDim;

  auto ldcDim = getLeadingDim(memrefC);
  if (failed(ldcDim)) {
    LLVM_DEBUG(llvm::dbgs() << "Cannot compute ldc\n");
    return failure();
  }
  int64_t ldc = *ldcDim;

  // If we are dealing with a BRGEMM we need to pass two extra dimensions:
  // - strideA and strideB that represent the stride between different GEMM
  // in BRGEMM.
  if (isBrgemm) {
    int64_t strideA = lda * m;
    int64_t strideB = ldb * k;
    return DenseI64ArrayAttr::get(
        rewriter.getContext(),
        ArrayRef<int64_t>{m, n, k, lda, ldb, ldc, strideA, strideB});
  }
  return DenseI64ArrayAttr::get(rewriter.getContext(),
                                ArrayRef<int64_t>{m, n, k, lda, ldb, ldc});
}

template <typename OpTy>
static ArrayAttr getGemmFlags(RewriterBase &rewriter, OpTy opTy) {
  auto memrefB = opTy.getMemRefInputType(1);
  xsmm::GemmFlagsAttr gemmFlag =
      (vnni::utils::isInVnniLayout(memrefB))
          ? xsmm::GemmFlagsAttr::get(rewriter.getContext(),
                                     xsmm::GemmFlags::VNNI_B)
          : xsmm::GemmFlagsAttr::get(rewriter.getContext(),
                                     xsmm::GemmFlags::NONE);
  return rewriter.getArrayAttr(gemmFlag);
}

struct ConvertTppGemmOp : public OpRewritePattern<tpp::GemmOp> {
  using OpRewritePattern<tpp::GemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpp::GemmOp matmulOp,
                                PatternRewriter &rewriter) const override {
    if (!matmulOp.hasBufferSemantics()) {
      return rewriter.notifyMatchFailure(matmulOp,
                                         "xsmm expects buffer semantics");
    }

    Location loc = matmulOp.getLoc();
    auto dims = getSizesAndLeadingDimsForGemmLikeOp(rewriter, matmulOp);
    if (failed(dims)) {
      return rewriter.notifyMatchFailure(
          matmulOp, "Cannot compute leading dims or sizes");
    }

    auto dtype = xsmm::utils::getDataType(rewriter, matmulOp.getOutputType());
    IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
    Value dispatched = rewriter.create<xsmm::GemmDispatchOp>(
        loc, integer64, *dims, getGemmFlags(rewriter, matmulOp), dtype);

    SmallVector<Value, 6> invokeOperands;
    invokeOperands.push_back(dispatched);
    invokeOperands.append(matmulOp->getOperands().begin(),
                          matmulOp->getOperands().end());
    // Drop the aliasing output operand.
    invokeOperands.pop_back();
    rewriter.replaceOpWithNewOp<xsmm::GemmOp>(matmulOp, dtype, invokeOperands);
    return success();
  }
};

struct ConvertTppBrgemmOp : public OpRewritePattern<tpp::BrgemmOp> {
  using OpRewritePattern<tpp::BrgemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpp::BrgemmOp brgemmOp,
                                PatternRewriter &rewriter) const override {
    if (!brgemmOp.hasBufferSemantics()) {
      return rewriter.notifyMatchFailure(brgemmOp,
                                         "xsmm expects buffer semantics");
    }

    Location loc = brgemmOp.getLoc();
    auto dims = getSizesAndLeadingDimsForGemmLikeOp(rewriter, brgemmOp);
    if (failed(dims)) {
      return rewriter.notifyMatchFailure(
          brgemmOp, "Cannot compute leading dims or sizes");
    }
    auto memrefB = brgemmOp.getMemRefInputType(1);
    int64_t batchSize = memrefB.getShape()[0];

    auto dtype = xsmm::utils::getDataType(rewriter, brgemmOp.getOutputType());
    IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);

    Value dispatched = rewriter.create<xsmm::BrgemmDispatchOp>(
        loc, integer64, *dims, getGemmFlags(rewriter, brgemmOp), dtype);

    Value batchDim = rewriter.create<arith::ConstantOp>(
        loc, integer64, rewriter.getIntegerAttr(integer64, batchSize));
    SmallVector<Value, 6> invokeOperands;
    invokeOperands.push_back(dispatched);
    invokeOperands.append(brgemmOp->getOperands().begin(),
                          brgemmOp->getOperands().end());
    // Drop the aliasing output operand.
    invokeOperands.pop_back();
    invokeOperands.push_back(batchDim);
    rewriter.replaceOpWithNewOp<xsmm::BrgemmOp>(brgemmOp, dtype,
                                                invokeOperands);
    return success();
  }
};

// Forward decl.
static xsmm::BinaryFlags getBinaryBCast(MemRefType operandType,
                                        MemRefType outputType,
                                        size_t operandNumber);

struct ConvertTppFusedBrgemmOp : public OpRewritePattern<tpp::FusedBrgemmOp> {
  using OpRewritePattern<tpp::FusedBrgemmOp>::OpRewritePattern;

  ArrayAttr getUnaryFlags(RewriterBase &rewriter,
                          tpp::FusedBrgemmOp brgemmOp) const {
    return rewriter.getArrayAttr(xsmm::UnaryFlagsAttr::get(
        rewriter.getContext(), xsmm::UnaryFlags::NONE));
  }

  ArrayAttr getBinaryFlags(RewriterBase &rewriter,
                           tpp::FusedBrgemmOp brgemmOp) const {
    auto binaryInputType =
        brgemmOp.getBiasOperand().getType().cast<MemRefType>();
    auto outputType = brgemmOp.getOutputType();
    return rewriter.getArrayAttr(xsmm::BinaryFlagsAttr::get(
        rewriter.getContext(),
        getBinaryBCast(binaryInputType, outputType, /*operandNumber=*/0)));
  }

  xsmm::BinaryKindAttr getBinaryKind(RewriterBase &rewriter,
                                     tpp::FusedBrgemmOp brgemmOp) const {
    auto kind = brgemmOp.getBinaryKind();
    auto ctx = rewriter.getContext();
    if (kind == tpp::FusedBinaryOpKind::NONE)
      return xsmm::BinaryKindAttr::get(ctx, xsmm::BinaryKind::NONE);
    if (kind == tpp::FusedBinaryOpKind::ADD)
      return xsmm::BinaryKindAttr::get(ctx, xsmm::BinaryKind::ADD);
    assert(false && "invalid binary kind");
  }

  xsmm::UnaryKindAttr getUnaryKind(RewriterBase &rewriter,
                                   tpp::FusedBrgemmOp brgemmOp) const {
    auto kind = brgemmOp.getUnaryKind();
    auto ctx = rewriter.getContext();
    if (kind == tpp::FusedUnaryOpKind::NONE)
      return xsmm::UnaryKindAttr::get(ctx, xsmm::UnaryKind::NONE);
    if (kind == tpp::FusedUnaryOpKind::RELU)
      return xsmm::UnaryKindAttr::get(ctx, xsmm::UnaryKind::RELU);
    assert(false && "invalid unary kind");
  }

  LogicalResult matchAndRewrite(tpp::FusedBrgemmOp brgemmOp,
                                PatternRewriter &rewriter) const override {
    if (!brgemmOp.hasBufferSemantics()) {
      return rewriter.notifyMatchFailure(brgemmOp,
                                         "xsmm expects buffer semantics");
    }

    Location loc = brgemmOp.getLoc();

    // Current limitation in LIBXSMM.
    // See: https://github.com/libxsmm/libxsmm/issues/766
    // Split into separate operations if bcast_col_in0 is not present when add
    // is fused.
    // TODO: remove the split once LIBXSMM is fixed.
    auto isBiasAdd = brgemmOp.getBinaryKind() == tpp::FusedBinaryOpKind::ADD;
    auto binaryFlag = getBinaryFlags(rewriter, brgemmOp)[0]
                          .cast<xsmm::BinaryFlagsAttr>()
                          .getValue();
    auto isBitSet = static_cast<uint64_t>(binaryFlag) &
                    static_cast<uint64_t>(xsmm::BinaryFlags::BCAST_COL_IN_0);
    if (isBiasAdd && !isBitSet)
      return tpp::utils::splitAndReplaceFusedOp(brgemmOp, rewriter);

    auto dims = getSizesAndLeadingDimsForGemmLikeOp(rewriter, brgemmOp);
    if (failed(dims)) {
      return rewriter.notifyMatchFailure(
          brgemmOp, "Cannot compute leading dims or sizes");
    }
    auto memrefB = brgemmOp.getMemRefInputType(1);
    int64_t batchSize = memrefB.getShape()[0];

    auto dtype = xsmm::utils::getDataType(rewriter, brgemmOp.getOutputType());
    IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);

    Value dispatched = rewriter.create<xsmm::FusedBrgemmDispatchOp>(
        loc, integer64, *dims, getBinaryKind(rewriter, brgemmOp),
        getUnaryKind(rewriter, brgemmOp), getGemmFlags(rewriter, brgemmOp),
        getUnaryFlags(rewriter, brgemmOp), getBinaryFlags(rewriter, brgemmOp),
        dtype);

    Value batchDim = rewriter.create<arith::ConstantOp>(
        loc, integer64, rewriter.getIntegerAttr(integer64, batchSize));
    SmallVector<Value, 6> invokeOperands;
    invokeOperands.push_back(dispatched);
    invokeOperands.append(brgemmOp->getOperands().begin(),
                          brgemmOp->getOperands().end());
    // Drop the aliasing output operand.
    invokeOperands.pop_back();
    invokeOperands.push_back(batchDim);
    rewriter.replaceOpWithNewOp<xsmm::FusedBrgemmOp>(brgemmOp, dtype,
                                                     invokeOperands);
    return success();
  }
};

// ======================================== Unary/Binary Ops Lowering

template <class OpKind, class OpFlags, class KindAttr, class FlagsAttr,
          class DispatchOp, class Op>
static LogicalResult lowerTPPtoXSMM(tpp::TppOp op, PatternRewriter &rewriter,
                                    Type elmTy, OpKind kind, OpFlags flags,
                                    ArrayRef<int64_t> dims) {
  auto *ctx = op.getContext();
  auto loc = op.getLoc();

  KindAttr kindAttr = KindAttr::get(ctx, kind);
  DenseI64ArrayAttr dimsAttr =
      DenseI64ArrayAttr::get(rewriter.getContext(), dims);
  auto flagsAttr = FlagsAttr::get(ctx, flags);
  IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
  xsmm::DataTypeAttr dtype =
      xsmm::utils::getDataType(rewriter, op.getOutputType());

  Value dispatched =
      rewriter.create<DispatchOp>(loc, integer64, kindAttr, dimsAttr,
                                  rewriter.getArrayAttr(flagsAttr), dtype);

  SmallVector<Value> invokeOperands;
  invokeOperands.push_back(dispatched);
  invokeOperands.append(op.getInputs().begin(), op.getInputs().end());
  invokeOperands.push_back(op.getOutput());

  rewriter.replaceOpWithNewOp<Op>(op, dtype, kindAttr, invokeOperands);
  return success();
}

static xsmm::UnaryFlags getUnaryFlags(tpp::TppOp tppOp) {
  Type inputType = tppOp.getInputs()[0].getType();

  // There are multiple ways to define a scalar.  f32, memref<1x1xf32> or
  // memref<f32>. Handle f32, and memref<1x1xf32>. memref<f32> is not allowed
  // in tpp at the moment.
  if (!inputType.isa<ShapedType>())
    return xsmm::UnaryFlags::BCAST_SCALAR;
  ArrayRef<int64_t> shapeInput = inputType.cast<ShapedType>().getShape();
  auto isOne = [](int64_t val) { return val == 1; };
  if (llvm::all_of(shapeInput, isOne))
    return xsmm::UnaryFlags::BCAST_SCALAR;

  Type outputType = tppOp.getOutputType();
  ArrayRef<int64_t> shapeOutput = outputType.cast<ShapedType>().getShape();
  assert(shapeOutput.size() >= shapeInput.size() &&
         "output rank must be >= input rank");
  SmallVector<int64_t> bShapeInput;
  computeBcastShapeInput(shapeOutput, shapeInput, bShapeInput);
  assert(shapeOutput.size() == bShapeInput.size());
  shapeInput = bShapeInput;

  if (shapeInput[1] == 1 && shapeOutput[1] > 1)
    return xsmm::UnaryFlags::BCAST_ROW;

  if (shapeInput[0] == 1 && shapeOutput[0] > 1)
    return xsmm::UnaryFlags::BCAST_COL;

  if (shapeInput[0] == shapeOutput[0] && shapeInput[1] == shapeOutput[1])
    return xsmm::UnaryFlags::NONE;

  assert(false && "failed to get bCast for tpp op");
}

static LogicalResult lowerUnaryTPPtoXSMM(PatternRewriter &rewriter,
                                         Operation *op, xsmm::UnaryKind kind) {
  auto tppOp = cast<tpp::TppOp>(op);
  if (!tppOp.hasBufferSemantics())
    return rewriter.notifyMatchFailure(tppOp, "xsmm expects a memref type");

  MemRefType outputMemRef = tppOp.getOutputType();
  int64_t m = outputMemRef.getShape()[0];
  int64_t n = outputMemRef.getShape()[1];
  auto ldo = getLeadingDim(outputMemRef);
  if (failed(ldo))
    return rewriter.notifyMatchFailure(tppOp, "cannot compute ldo");
  auto ldi = getLeadingDim(tppOp.getInputs()[0].getType());
  if (failed(ldi))
    return rewriter.notifyMatchFailure(tppOp, "cannot compute ldi");
  xsmm::UnaryFlags flags = getUnaryFlags(tppOp);
  return lowerTPPtoXSMM<xsmm::UnaryKind, xsmm::UnaryFlags, xsmm::UnaryKindAttr,
                        xsmm::UnaryFlagsAttr, xsmm::UnaryDispatchOp,
                        xsmm::UnaryOp>(tppOp, rewriter,
                                       outputMemRef.getElementType(), kind,
                                       flags, {m, n, *ldi, *ldo});
}

static LogicalResult lowerBinaryTPPtoXSMM(Operation *op,
                                          PatternRewriter &rewriter, Type elmTy,
                                          xsmm::BinaryKind kind,
                                          xsmm::BinaryFlags flags,
                                          ArrayRef<int64_t> dims) {
  assert(isa<tpp::TppOp>(op));
  auto tppOp = cast<tpp::TppOp>(op);
  if (!tppOp.hasBufferSemantics())
    return rewriter.notifyMatchFailure(tppOp, "xsmm expects a memref type");
  return lowerTPPtoXSMM<xsmm::BinaryKind, xsmm::BinaryFlags,
                        xsmm::BinaryKindAttr, xsmm::BinaryFlagsAttr,
                        xsmm::BinaryDispatchOp, xsmm::BinaryOp>(
      tppOp, rewriter, elmTy, kind, flags, dims);
}

struct ConvertTppIdentityOp : public OpRewritePattern<tpp::IdentityOp> {
  using OpRewritePattern<tpp::IdentityOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpp::IdentityOp identityOp,
                                PatternRewriter &rewriter) const override {
    return lowerUnaryTPPtoXSMM(rewriter, identityOp, xsmm::UnaryKind::IDENTITY);
  }
};

struct ConvertTppReluOp : public OpRewritePattern<tpp::ReluOp> {
  using OpRewritePattern<tpp::ReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpp::ReluOp reluOp,
                                PatternRewriter &rewriter) const override {
    return lowerUnaryTPPtoXSMM(rewriter, reluOp, xsmm::UnaryKind::RELU);
  }
};

struct ConvertTppZeroOp : public OpRewritePattern<tpp::ZeroOp> {
  using OpRewritePattern<tpp::ZeroOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpp::ZeroOp zeroOp,
                                PatternRewriter &rewriter) const override {
    return lowerUnaryTPPtoXSMM(rewriter, zeroOp, xsmm::UnaryKind::ZERO);
  }
};

// Given the operand type and the output type return the broadcast
// to use in the XSMM call.
static xsmm::BinaryFlags getBinaryBCast(MemRefType operandType,
                                        MemRefType outputType,
                                        size_t operandNumber) {

  enum class BCastType { NONE = 0, SCALAR, ROW, COL };

  auto shapeOutput = outputType.getShape();
  auto shapeOperand = operandType.getShape();
  assert(shapeOutput.size() >= shapeOperand.size() &&
         "Output rank must be >= operand rank");
  SmallVector<int64_t> bOperandShape;
  computeBcastShapeInput(shapeOutput, shapeOperand, bOperandShape);
  assert(shapeOutput.size() == bOperandShape.size());
  assert(shapeOutput.size() == 2);

  auto getBCastEnum = [](BCastType bCastType,
                         std::optional<unsigned> operand) -> xsmm::BinaryFlags {
    switch (bCastType) {
    case BCastType::NONE:
      return xsmm::BinaryFlags::NONE;
    case BCastType::SCALAR:
      assert(operand != std::nullopt && "Require operand idx");
      assert(*operand == 1 || *operand == 0 && "Expect idx to be 1 or 0");
      if (*operand == 0)
        return xsmm::BinaryFlags::BCAST_SCALAR_IN_0;
      else
        return xsmm::BinaryFlags::BCAST_SCALAR_IN_1;
    case BCastType::ROW:
      assert(operand != std::nullopt && "Require operand idx");
      assert(*operand == 1 || *operand == 0 && "Expect idx to be 1 or 0");
      if (*operand == 0)
        return xsmm::BinaryFlags::BCAST_ROW_IN_0;
      else
        return xsmm::BinaryFlags::BCAST_ROW_IN_1;
    case BCastType::COL:
      assert(operand != std::nullopt && "Require operand idx");
      assert(*operand == 1 || *operand == 0 && "Expect idx to be 1 or 0");
      if (*operand == 0)
        return xsmm::BinaryFlags::BCAST_COL_IN_0;
      else
        return xsmm::BinaryFlags::BCAST_COL_IN_1;
    }
    assert(false && "unrechable");
  };

  // Multiple way to define a scalar. Check if the memref
  // is a scalar here.
  auto isOne = [](int64_t val) { return val == 1; };
  if (llvm::all_of(bOperandShape, isOne))
    return getBCastEnum(BCastType::SCALAR, operandNumber);

  if (bOperandShape[1] == 1 && shapeOutput[1] > 1)
    return getBCastEnum(BCastType::ROW, operandNumber);
  if (bOperandShape[0] == 1 && shapeOutput[0] > 1)
    return getBCastEnum(BCastType::COL, operandNumber);
  if (bOperandShape == shapeOutput)
    return getBCastEnum(BCastType::NONE, operandNumber);

  assert(false && "failed to get bCast for tpp.add");
}

struct ConvertTppAddOp : public OpRewritePattern<tpp::AddOp> {
  using OpRewritePattern<tpp::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpp::AddOp addOp,
                                PatternRewriter &rewriter) const override {
    if (!addOp.hasBufferSemantics())
      return rewriter.notifyMatchFailure(addOp, "xsmm expects a memref type");

    MemRefType outputMemRef = addOp.getOutputType();
    assert(outputMemRef.getRank() == 2 && "expect rank 2 for TPP ops");

    int64_t m = outputMemRef.getShape()[0];
    int64_t n = outputMemRef.getShape()[1];

    auto lhsMemRef = addOp.getInputs()[0].getType().cast<MemRefType>();
    auto rhsMemRef = addOp.getInputs()[1].getType().cast<MemRefType>();

    auto ldiLhsDim = getLeadingDim(lhsMemRef);
    if (failed(ldiLhsDim))
      return rewriter.notifyMatchFailure(addOp, "Cannot compute ldi on lhs");
    int64_t ldiLhs = *ldiLhsDim;

    auto ldiRhsDim = getLeadingDim(rhsMemRef);
    if (failed(ldiRhsDim))
      return rewriter.notifyMatchFailure(addOp, "Cannot compute ldi on rhs");
    int64_t ldiRhs = *ldiRhsDim;

    auto ldoDim = getLeadingDim(outputMemRef);
    if (failed(ldoDim))
      return rewriter.notifyMatchFailure(addOp, "Cannot compute ldo");
    int64_t ldo = *ldoDim;

    xsmm::BinaryFlags bCastOnLhs = getBinaryBCast(lhsMemRef, outputMemRef, 0);
    xsmm::BinaryFlags bCastOnRhs = getBinaryBCast(rhsMemRef, outputMemRef, 1);

    LLVM_DEBUG(llvm::dbgs() << stringifyBinaryFlags(bCastOnLhs) << "\n");
    LLVM_DEBUG(llvm::dbgs() << stringifyBinaryFlags(bCastOnRhs) << "\n");

    xsmm::BinaryFlags bCast =
        (bCastOnLhs != xsmm::BinaryFlags::NONE) ? bCastOnLhs : bCastOnRhs;

    return lowerBinaryTPPtoXSMM(addOp, rewriter, outputMemRef.getElementType(),
                                xsmm::BinaryKind::ADD, bCast,
                                {m, n, ldiLhs, ldiRhs, ldo});
  }
};

struct ConvertTppToXsmm : public ConvertTppToXsmmBase<ConvertTppToXsmm> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertTppIdentityOp, ConvertTppReluOp, ConvertTppZeroOp,
                 ConvertTppAddOp, ConvertTppGemmOp, ConvertTppBrgemmOp,
                 ConvertTppFusedBrgemmOp>(patterns.getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createConvertTppToXsmmPass() {
  return std::make_unique<ConvertTppToXsmm>();
}
