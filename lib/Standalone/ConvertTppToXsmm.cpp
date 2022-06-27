//===- ConvertTppToXsmm.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/Tpp/TppOps.h"
#include "Standalone/Dialect/Xsmm/XsmmOps.h"
#include "Standalone/Passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "Standalone/Passes.h.inc"

// TODO: Xsmm should take a I64EnumAttr not a FlatSymbolRefAttr.
// TODO: Nice if we can mirror typedef in LIBXSMM instead of
// passing around i32.
namespace {

struct ConvertTppMatmulOp : public OpRewritePattern<MatmulOp> {
  using OpRewritePattern<MatmulOp>::OpRewritePattern;

  Attribute getIntAttr(Builder &builder, IntegerType tp, int64_t val) const {
    return builder.getIntegerAttr(tp, APInt(tp.getWidth(), val));
  }

  LogicalResult matchAndRewrite(MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    Location loc = matmulOp.getLoc();
    FlatSymbolRefAttr attrDispatch =
        FlatSymbolRefAttr::get(matmulOp.getContext(), "xsmm_matmul_dispatch");
    MemRefType memrefC = matmulOp.getMatrixCType();
    MemRefType memrefA = matmulOp.getMatrixAType();
    int64_t m = memrefC.getShape()[0];
    int64_t n = memrefC.getShape()[1];
    int64_t k = memrefA.getShape()[1];
    int64_t lda = m;
    int64_t ldb = k;
    int64_t ldc = m;
    SmallVector<Value, 6> dispatchOperands;
    IntegerType integer = IntegerType::get(rewriter.getContext(), 32);
    IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
    dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
        loc, integer, getIntAttr(rewriter, integer, m)));
    dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
        loc, integer, getIntAttr(rewriter, integer, n)));
    dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
        loc, integer, getIntAttr(rewriter, integer, k)));
    dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
        loc, integer, getIntAttr(rewriter, integer, lda)));
    dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
        loc, integer, getIntAttr(rewriter, integer, ldb)));
    dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
        loc, integer, getIntAttr(rewriter, integer, ldc)));
    Value dispatched = rewriter.create<xsmm::DispatchOp>(
        loc, integer64, attrDispatch, dispatchOperands);

    SmallVector<Value, 6> invokeOperands;
    invokeOperands.push_back(dispatched);
    invokeOperands.append(matmulOp->getOperands().begin(),
                          matmulOp->getOperands().end());
    FlatSymbolRefAttr attrInvoke =
        FlatSymbolRefAttr::get(matmulOp.getContext(), "xsmm_matmul_invoke");
    rewriter.replaceOpWithNewOp<xsmm::TernaryOp>(matmulOp, attrInvoke,
                                                 invokeOperands);
    return success();
  }
};

struct ConvertTppIdentityOp : public OpRewritePattern<IdentityOp> {
  using OpRewritePattern<IdentityOp>::OpRewritePattern;

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
        llvm_unreachable("bCast semantics for identity op broken");
    }
  }

  // Return ldi and bCast.
  std::pair<int64_t, int64_t> getLdiAndBCast(IdentityOp identityOp,
                                             int64_t ldo) const {
    Type inputType = identityOp.getInput().getType();
    if (!inputType.isa<ShapedType>()) {
      int64_t bCast = 3; // scalar broadcast
      int64_t ldi = 1;
      return {ldi, bCast};
    }

    ArrayRef<int64_t> shapeInput =
        identityOp.getInput().getType().cast<ShapedType>().getShape();
    ArrayRef<int64_t> shapeOutput =
        identityOp.getOutput().getType().cast<ShapedType>().getShape();
    assert(shapeOutput.size() >= shapeInput.size() &&
           "output rank must be >= input rank");
    SmallVector<int64_t, 4> bShapeInput;
    computeBcastShapeInput(shapeOutput, shapeInput, bShapeInput);
    assert(shapeOutput.size() == bShapeInput.size());

    if (shapeInput[1] == 1 && shapeOutput[1] > 1) {
      int64_t bCast = 1; // row broadcast
      int64_t ldi = shapeInput[0];
      return {ldi, bCast};
    }

    if (shapeInput[0] == 1 && shapeOutput[0] > 1) {
      int64_t bCast = 2; // col broadcast
      int64_t ldi = shapeInput[1];
      return {ldi, bCast};
    }

    if (shapeInput[0] == shapeOutput[0] && shapeInput[1] == shapeOutput[1]) {
      int64_t bCast = 0; // no broadcast
      int64_t ldi = shapeInput[0];
      return {ldi, bCast};
    }
    // TODO: Handle memref<1x1xf32> and memref<f32>
    llvm_unreachable("failed to get ldi and bCast for identity");
  }

  // See:
  // https://github.com/libxsmm/libxsmm/blob/d4cd3b2fe127a32562a9cd248fc2f8d97eeaa078/include/libxsmm_typedefs.h#L146
  // TODO: We should duplicate the typedef in xsmm or import "libxsmm.h".
  // I prefer the former.
  int32_t convertDataType(Type t) const {
    if (t.isa<ShapedType>())
      t = t.cast<ShapedType>().getElementType();
    if (t.isF32())
      return 1;
    llvm_unreachable("only f32");
  }

  // TODO: method repeated multiple times.
  Attribute getIntAttr(Builder &builder, IntegerType tp, int64_t val) const {
    return builder.getIntegerAttr(tp, APInt(tp.getWidth(), val));
  }

  LogicalResult matchAndRewrite(IdentityOp identityOp,
                                PatternRewriter &rewriter) const override {
    Location loc = identityOp.getLoc();
    FlatSymbolRefAttr attrDispatch = FlatSymbolRefAttr::get(
        identityOp.getContext(), "xsmm_identity_dispatch");
    // no conversion if identity is a scalar operation.
    Type outputType = identityOp.getOutput().getType();
    if (!outputType.isa<ShapedType>())
      return failure();

    Type inputType = identityOp.getInput().getType();
    MemRefType outputMemRef = outputType.cast<MemRefType>();
    int64_t m = outputMemRef.getShape()[0];
    int64_t n = outputMemRef.getShape()[1];
    int64_t ldo = n;
    std::pair<int64_t, int64_t> ldiAndBCast = getLdiAndBCast(identityOp, ldo);
    int64_t ldi = ldiAndBCast.first;
    int64_t bCast = ldiAndBCast.second;
    SmallVector<Value, 8> dispatchOperands;
    IntegerType integer = IntegerType::get(rewriter.getContext(), 32);
    IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
    dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
        loc, integer, getIntAttr(rewriter, integer, m)));
    dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
        loc, integer, getIntAttr(rewriter, integer, n)));
    dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
        loc, integer, getIntAttr(rewriter, integer, ldi)));
    dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
        loc, integer, getIntAttr(rewriter, integer, ldo)));
    dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
        loc, integer,
        getIntAttr(rewriter, integer, convertDataType(inputType))));
    dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
        loc, integer,
        getIntAttr(rewriter, integer, convertDataType(outputType))));
    // TODO: Clarify the "compute type". Use output type for now.
    dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
        loc, integer,
        getIntAttr(rewriter, integer, convertDataType(outputType))));
    dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
        loc, integer, getIntAttr(rewriter, integer, bCast)));
    Value dispatched = rewriter.create<xsmm::DispatchOp>(
        loc, integer64, attrDispatch, dispatchOperands);

    SmallVector<Value, 6> invokeOperands;
    invokeOperands.push_back(dispatched);
    invokeOperands.append(identityOp->getOperands().begin(),
                          identityOp->getOperands().end());
    FlatSymbolRefAttr attrInvoke =
        FlatSymbolRefAttr::get(identityOp.getContext(), "xsmm_identity_invoke");

    rewriter.replaceOpWithNewOp<xsmm::UnaryOp>(identityOp, attrInvoke,
                                               invokeOperands);
    return success();
  }
};

void populateTppToXsmmPatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<ConvertTppIdentityOp,
               ConvertTppMatmulOp>(patterns.getContext());
  // clang-format on
}

struct ConvertTppToXsmm : public ConvertTppToXsmmBase<ConvertTppToXsmm> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateTppToXsmmPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createConvertTppToXsmmPass() {
  return std::make_unique<ConvertTppToXsmm>();
}
