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
#include "TPP/Transforms/Transforms.h"
#include "TPP/Transforms/Utils/VNNIUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_CONVERTTPPTOXSMM
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

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
  auto expectedVNNIRank =
      (isBrgemm) ? vnni::utils::VnniOp::BRGEMM_INS : vnni::utils::VnniOp::GEMM;
  int64_t ldb = (vnni::utils::isInVnniLayout(expectedVNNIRank, memrefB))
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
  static_assert(llvm::is_one_of<OpTy, tpp::GemmOp, tpp::BrgemmOp,
                                tpp::FusedBrgemmOp>::value);

  bool isBrgemm = std::is_same<OpTy, tpp::BrgemmOp>::value ||
                  std::is_same<OpTy, tpp::FusedBrgemmOp>::value;
  auto expectedVnniRank =
      (isBrgemm) ? vnni::utils::VnniOp::BRGEMM_INS : vnni::utils::VnniOp::GEMM;
  auto memrefB = opTy.getMemRefInputType(1);
  xsmm::GemmFlagsAttr gemmFlag =
      (vnni::utils::isInVnniLayout(expectedVnniRank, memrefB))
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
    auto flags = xsmm::utils::getBinaryFlags(binaryInputType, outputType,
                                             /*operandNumber=*/0);
    assert(succeeded(flags));
    return rewriter.getArrayAttr(
        xsmm::BinaryFlagsAttr::get(rewriter.getContext(), *flags));
  }

  xsmm::BinaryKindAttr getBinaryKind(RewriterBase &rewriter,
                                     tpp::FusedBrgemmOp brgemmOp) const {
    auto kind = brgemmOp.getBinaryKind();
    auto *ctx = rewriter.getContext();
    if (kind == tpp::FusedBinaryOpKind::NONE)
      return xsmm::BinaryKindAttr::get(ctx, xsmm::BinaryKind::NONE);
    if (kind == tpp::FusedBinaryOpKind::ADD)
      return xsmm::BinaryKindAttr::get(ctx, xsmm::BinaryKind::ADD);
    assert(false && "invalid binary kind");
  }

  xsmm::UnaryKindAttr getUnaryKind(RewriterBase &rewriter,
                                   tpp::FusedBrgemmOp brgemmOp) const {
    auto kind = brgemmOp.getUnaryKind();
    auto *ctx = rewriter.getContext();
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
  auto flags = xsmm::utils::getUnaryFlags(tppOp.getInputs()[0].getType(),
                                          tppOp.getOutputType());
  if (failed(flags))
    return failure();
  return lowerTPPtoXSMM<xsmm::UnaryKind, xsmm::UnaryFlags, xsmm::UnaryKindAttr,
                        xsmm::UnaryFlagsAttr, xsmm::UnaryDispatchOp,
                        xsmm::UnaryOp>(tppOp, rewriter,
                                       outputMemRef.getElementType(), kind,
                                       *flags, {m, n, *ldi, *ldo});
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

    auto bCastOnLhs = xsmm::utils::getBinaryFlags(lhsMemRef, outputMemRef, 0);
    auto bCastOnRhs = xsmm::utils::getBinaryFlags(rhsMemRef, outputMemRef, 1);
    if (failed(bCastOnLhs) || failed(bCastOnRhs))
      return failure();

    LLVM_DEBUG(llvm::dbgs() << stringifyBinaryFlags(*bCastOnLhs) << "\n");
    LLVM_DEBUG(llvm::dbgs() << stringifyBinaryFlags(*bCastOnRhs) << "\n");

    xsmm::BinaryFlags bCast =
        (bCastOnLhs != xsmm::BinaryFlags::NONE) ? *bCastOnLhs : *bCastOnRhs;

    return lowerBinaryTPPtoXSMM(addOp, rewriter, outputMemRef.getElementType(),
                                xsmm::BinaryKind::ADD, bCast,
                                {m, n, ldiLhs, ldiRhs, ldo});
  }
};

struct ConvertTppToXsmm
    : public tpp::impl::ConvertTppToXsmmBase<ConvertTppToXsmm> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertTppIdentityOp, ConvertTppReluOp, ConvertTppZeroOp,
                 ConvertTppAddOp, ConvertTppGemmOp, ConvertTppBrgemmOp,
                 ConvertTppFusedBrgemmOp>(patterns.getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace
