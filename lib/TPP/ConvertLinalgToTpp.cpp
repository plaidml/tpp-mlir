//===- LinalgConvertToTpp.cpp ------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Tpp/TppOps.h"
#include "TPP/Dialect/Tpp/TppUtils.h"
#include "TPP/Passes.h"
#include "TPP/TransformUtils.h"
#include "TPP/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

#define DEBUG_TYPE "linalg-convert-to-tpp"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

namespace {

// Convert a linalg.generic to a tpp operation.
struct ConvertGenericOpToTpp : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult rewriteToTppOp(linalg::GenericOp linalgOp,
                               PatternRewriter &rewriter) const {
    SmallVector<Value> operands;
    if (tpp::utils::isTppZero(linalgOp, &operands)) {
      assert(operands.size() == 1 && "tpp.zero expects one operand");
      rewriter.replaceOpWithNewOp<tpp::ZeroOp>(linalgOp, operands[0],
                                               operands[0]);
      return success();
    }

    if (tpp::utils::isTppIdentity(linalgOp, &operands)) {
      assert(operands.size() == 2 && "tpp.identity expects two operands");
      rewriter.replaceOpWithNewOp<tpp::IdentityOp>(linalgOp, operands[0],
                                                   operands[1]);
      return success();
    }

    if (tpp::utils::isTppRelu(linalgOp, &operands)) {
      assert(operands.size() == 2 && "tpp.relu expects two operands");
      rewriter.replaceOpWithNewOp<tpp::ReluOp>(linalgOp, operands[0],
                                               operands[1]);
      return success();
    }

    if (tpp::utils::isTppAdd(linalgOp, &operands)) {
      assert(operands.size() == 3 && "tpp.add expects three operands");
      rewriter.replaceOpWithNewOp<tpp::AddOp>(
          linalgOp, ValueRange{operands[0], operands[1]}, operands[2]);
      return success();
    }

    if (linalgx::utils::isMatmulOp(linalgOp, &operands)) {
      assert(operands.size() == 3 && "tpp.matmul expects three operands");
      rewriter.replaceOpWithNewOp<tpp::GemmOp>(
          linalgOp, ValueRange{operands[0], operands[1], operands[2]},
          operands[2]);
      return success();
    }

    return rewriter.notifyMatchFailure(
        linalgOp, "failed to match to a known tpp operation");
  }

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!tpp::utils::hasStaticShape(linalgOp))
      return rewriter.notifyMatchFailure(
          linalgOp, "Expect static shape when mapping to tpp");
    return rewriteToTppOp(linalgOp, rewriter);
  }
};

// Convert a linalg.batch_reduce_matmul to a tpp.brgemm.
struct ConvertBrgemmToTpp
    : public OpRewritePattern<linalg::BatchReduceMatmulOp> {
  using OpRewritePattern<linalg::BatchReduceMatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::BatchReduceMatmulOp brMatmulOp,
                                PatternRewriter &rewriter) const override {
    if (!tpp::utils::hasStaticShape(brMatmulOp))
      return rewriter.notifyMatchFailure(
          brMatmulOp, "Expect static shape when mapping to tpp");
    SmallVector<Value> inputs = brMatmulOp.getDpsInputOperands();
    inputs.push_back(brMatmulOp.getDpsInitOperands()[0]->get());
    SmallVector<Value> outputs = brMatmulOp.getDpsInitOperands();
    rewriter.replaceOpWithNewOp<tpp::BrgemmOp>(brMatmulOp, inputs, outputs[0]);
    return success();
  }
};

// Convert a linalg.matmul to a tpp.matmul.
struct ConvertMatmulToTpp : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    if (!tpp::utils::hasStaticShape(matmulOp))
      return rewriter.notifyMatchFailure(
          matmulOp, "Expect static shape when mapping to tpp");
    SmallVector<Value> inputs = matmulOp.getDpsInputOperands();
    inputs.push_back(matmulOp.getDpsInitOperands()[0]->get());
    SmallVector<Value> outputs = matmulOp.getDpsInitOperands();
    rewriter.replaceOpWithNewOp<tpp::GemmOp>(matmulOp, inputs, outputs[0]);
    return success();
  }
};

// Convert a linalg.fill to a tpp.zero.
struct ConvertFillToTpp : public OpRewritePattern<linalg::FillOp> {
  using OpRewritePattern<linalg::FillOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::FillOp fillOp,
                                PatternRewriter &rewriter) const override {
    if (!tpp::utils::hasStaticShape(fillOp))
      return rewriter.notifyMatchFailure(
          fillOp, "Expect static shape when mapping to tpp");

    auto inputs = fillOp.getInputs();
    if (!tpp::utils::isZeroTensor(inputs[0]))
      return rewriter.notifyMatchFailure(fillOp, "Unsupported fill type");

    auto outputs = fillOp.getOutputs();
    if (outputs[0].getType().cast<ShapedType>().getRank() > 2)
      return rewriter.notifyMatchFailure(fillOp,
                                         "Expect output rank at most 2");

    rewriter.replaceOpWithNewOp<tpp::ZeroOp>(fillOp, outputs[0], outputs[0]);
    return success();
  }
};

struct ConvertLinalgToTpp : public ConvertLinalgToTppBase<ConvertLinalgToTpp> {
  ConvertLinalgToTpp() = default;
  void runOnOperation() override {
    MLIRContext *ctx = getOperation().getContext();
    RewritePatternSet patterns(ctx);
    tpp::populateConvertLinalgToTppPatterns(patterns);
    memref::SubViewOp::getCanonicalizationPatterns(patterns, ctx);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // end namespace

void mlir::tpp::populateConvertLinalgToTppPatterns(
    RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<ConvertGenericOpToTpp,
               ConvertBrgemmToTpp,
               ConvertMatmulToTpp,
               ConvertFillToTpp>(patterns.getContext());
  // clang-format on
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createConvertLinalgToTppPass() {
  return std::make_unique<ConvertLinalgToTpp>();
}
