//===- ConvertLinalgToXsmm.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Tpp/TppUtils.h"
#include "TPP/Dialect/Xsmm/XsmmOps.h"
#include "TPP/Dialect/Xsmm/XsmmUtils.h"
#include "TPP/Passes.h"
#include "TPP/TransformUtils.h"
#include "TPP/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

#define DEBUG_TYPE "convert-linalg-to-xsmm"

namespace {

struct ConvertLinalgToXsmm
    : public ConvertLinalgToXsmmBase<ConvertLinalgToXsmm> {
  void runOnOperation() override;
};

namespace {
struct UnaryInfo {
  unsigned m;
  unsigned n;

  int64_t ldi;
  int64_t ldo;
};
} // namespace

// Check if the strides associated with `operand` are valid strides
// for XSMM: Strides must be statically known.
static FailureOr<SmallVector<int64_t>> verifyStrides(Type operandType) {
  assert(!isa<RankedTensorType>(operandType));

  // Scalar type.
  if (!isa<MemRefType>(operandType))
    return SmallVector<int64_t>{1};

  // MemRef type.
  auto memref = cast<MemRefType>(operandType);
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(getStridesAndOffset(memref, strides, offset))) {
    return failure();
  }
  if (llvm::any_of(strides, [](int64_t stride) {
        return stride == ShapedType::kDynamic;
      })) {
    return failure();
  }
  if (strides.back() != 1)
    return failure();
  return strides;
}

// Return true if all the operand have the same type. No implicit conversion in
// the linalgOp.
static bool hasEqualTypes(linalg::LinalgOp linalgOp) {
  // assert(linalgOp.getNumInitOperands > 0);
  OpOperand *outputOperand = linalgOp.getDpsInitOperands().back();
  auto elemType = getElementTypeOrSelf(outputOperand->get().getType());

  if (!llvm::all_of(linalgOp.getDpsInitOperands(), [&](OpOperand *operand) {
        auto currentOperandType =
            getElementTypeOrSelf(operand->get().getType());
        return currentOperandType == elemType;
      })) {
    return false;
  }

  return llvm::all_of(linalgOp.getDpsInputOperands(), [&](OpOperand *operand) {
    auto currentOperandType = getElementTypeOrSelf(operand->get().getType());
    return currentOperandType == elemType;
  });
}

// Replace `linalgOp` with a unary dispatch plus invoke.
static void replaceOpWithUnary(RewriterBase &rewriter,
                               linalg::LinalgOp linalgOp, UnaryInfo unaryInfo,
                               ArrayAttr flags, xsmm::UnaryKindAttr kind) {
  Location loc = linalgOp.getLoc();
  IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
  DenseI64ArrayAttr dims = DenseI64ArrayAttr::get(
      rewriter.getContext(), ArrayRef<int64_t>{unaryInfo.m, unaryInfo.n,
                                               unaryInfo.ldi, unaryInfo.ldo});
  auto dtype = xsmm::utils::getDataType(
      rewriter, linalgOp.getDpsInitOperands()[0]->get().getType());
  Value dispatched = rewriter.create<xsmm::UnaryDispatchOp>(
      loc, integer64, kind, dims, flags, dtype);
  SmallVector<Value> invokeOperands;
  invokeOperands.push_back(dispatched);
  invokeOperands.append(linalgOp->getOperands().begin(),
                        linalgOp->getOperands().end());
  rewriter.replaceOpWithNewOp<xsmm::UnaryOp>(linalgOp, dtype, kind,
                                             invokeOperands);
}

// Convert a linalg.fill to XSMM zero, if the fill fills with zeros.
struct ConvertFillOpToUnaryZero : public OpRewritePattern<linalg::FillOp> {
  using OpRewritePattern<linalg::FillOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::FillOp fillOp,
                                PatternRewriter &rewriter) const override {
    if (!fillOp.hasBufferSemantics() || fillOp.hasDynamicShape() ||
        !hasEqualTypes(fillOp)) {
      return failure();
    }
    auto input = fillOp.getDpsInputOperands()[0];
    if (!tpp::utils::isZeroTensor(input->get()))
      return failure();

    auto output = fillOp.getDpsInitOperands()[0];
    ShapedType outputType = output->get().getType().cast<ShapedType>();
    auto outputRank = outputType.getRank();
    if (outputRank != 2)
      return failure();

    // Verify strides and minor dimensions.
    auto stridesOnOutput = verifyStrides(outputType);
    if (failed(stridesOnOutput))
      return failure();

    UnaryInfo unaryInfo;
    unaryInfo.m = outputType.getShape()[0];
    unaryInfo.n = outputType.getShape()[1];
    unaryInfo.ldo = stridesOnOutput->front();
    // fillOp has a scalar input.
    unaryInfo.ldi = 1;

    auto flags = rewriter.getArrayAttr(xsmm::UnaryFlagsAttr::get(
        rewriter.getContext(), xsmm::UnaryFlags::BCAST_SCALAR));
    xsmm::UnaryKindAttr kind =
        xsmm::UnaryKindAttr::get(rewriter.getContext(), xsmm::UnaryKind::ZERO);
    replaceOpWithUnary(rewriter, fillOp, unaryInfo, flags, kind);
    return success();
  }
};

void ConvertLinalgToXsmm::runOnOperation() {
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  tpp::populateLinalgToXsmmPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}
} // namespace

void mlir::tpp::populateLinalgToXsmmPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertFillOpToUnaryZero>(patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createConvertLinalgToXsmmPass() {
  return std::make_unique<ConvertLinalgToXsmm>();
}
