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

  LogicalResult matchAndRewrite(IdentityOp identityOp,
                                PatternRewriter &rewriter) const override {
    Location loc = identityOp.getLoc();
    FlatSymbolRefAttr attrInvoke =
        FlatSymbolRefAttr::get(identityOp.getContext(), "xsmm_identity_invoke");
    // no conversion if identity is a scalar operation.
    Type outputType = identityOp.getOutput().getType();
    if (!outputType.isa<ShapedType>())
      return failure();
    // MemRefType outputMemRef = outputType.cast<MemRefType>();
    // int64_t m = outputMemRef.getShape()[0];
    // int64_t n = outputMemRef.getShape()[1];

    rewriter.replaceOpWithNewOp<xsmm::UnaryOp>(identityOp, attrInvoke,
                                               identityOp->getOperands());
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
