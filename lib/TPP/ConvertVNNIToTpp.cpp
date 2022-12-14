//===- ConvertVNNIToTpp.cpp ----------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "TPP/Dialect/Tpp/TppOps.h"
#include "TPP/Dialect/VNNI/VNNIOps.h"
#include "TPP/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

struct ConvertMatmulOp : public OpRewritePattern<vnni::MatmulOp> {
  using OpRewritePattern<vnni::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vnni::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    if (!matmulOp.hasBufferSemantics()) {
      return rewriter.notifyMatchFailure(
          matmulOp, "Cannot lower vnni matmul to tpp, op not bufferized");
    }
    if (matmulOp.hasDynamicShape()) {
      return rewriter.notifyMatchFailure(
          matmulOp, "Cannot lower vnni matmul to tpp, op has dynamic shape");
    }
    rewriter.replaceOpWithNewOp<tpp::VNNIMatmulOp>(
        matmulOp, matmulOp.getMatrixA(), matmulOp.getMatrixB(),
        matmulOp.getMatrixC());
    return success();
  }
};

struct ConvertBRGemmOp : public OpRewritePattern<vnni::BRGemmOp> {
  using OpRewritePattern<vnni::BRGemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vnni::BRGemmOp brgemmOp,
                                PatternRewriter &rewriter) const override {

    if (!brgemmOp.hasBufferSemantics()) {
      return rewriter.notifyMatchFailure(
          brgemmOp, "Cannot lower vnni brgemm to tpp, op not bufferized");
    }
    if (brgemmOp.hasDynamicShape()) {
      return rewriter.notifyMatchFailure(
          brgemmOp, "Cannot lower vnni brgemm to tpp, op has dynamic shape");
    }
    rewriter.replaceOpWithNewOp<tpp::VNNIBrgemmOp>(
        brgemmOp, brgemmOp.getMatrixA(), brgemmOp.getMatrixB(),
        brgemmOp.getMatrixC());
    return success();
  }
};

void populateVNNIToTppPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertMatmulOp, ConvertBRGemmOp>(patterns.getContext());
}

struct ConvertVNNIToTpp : public ConvertVNNIToTppBase<ConvertVNNIToTpp> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateVNNIToTppPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::tpp::createConvertVNNIToTppPass() {
  return std::make_unique<ConvertVNNIToTpp>();
}
