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

// Given an operand 'operand' check if it is a scalar
// or a static shape type with rank <= 2.
static LogicalResult checkOperandForTpp(Value operand) {
  Type operandType = operand.getType();
  if (!operandType.isa<ShapedType>())
    return success();
  if (auto shapedType = operandType.dyn_cast_or_null<ShapedType>()) {
    if (!shapedType.hasStaticShape())
      return failure();
    unsigned rank = shapedType.getRank();
    if (rank == 0 || rank > 2)
      return failure();
  }
  return success();
}

// Convert a linalg.generic to a tpp operation.
struct ConvertGenericOpToTpp : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult rewriteToTppOp(linalg::GenericOp linalgOp,
                               ArrayRef<Value> operands, IRMapping &mapping,
                               PatternRewriter &rewriter) const {
    tpp::utils::OperandInfo operandInfo;
    if (tpp::utils::isTppIdentity(linalgOp)) {
      assert(operands.size() == 2 && "Expect two operands");
      // FIXME: We are not converting 1d to xsmm. See #375.
      Type outputType = operands[1].getType();
      if (outputType.cast<ShapedType>().getRank() != 2)
        return failure();
      rewriter.replaceOpWithNewOp<tpp::IdentityOp>(linalgOp, operands[0],
                                                   operands[1]);
      return success();
    }
    if (tpp::utils::isTppRelu(linalgOp, operandInfo)) {
      assert(operandInfo.inputs.size() == 1);
      assert(operandInfo.outputs.size() == 1);
      rewriter.replaceOpWithNewOp<tpp::ReluOp>(
          linalgOp, mapping.lookup(operandInfo.inputs[0]),
          mapping.lookup(operandInfo.outputs[0]));
      return success();
    }
    if (tpp::utils::isTppAdd(linalgOp)) {
      // Allow either:
      // 1. A = A + B
      // 2. C = A + B
      Value output =
          (linalgOp.getNumOperands() == 3) ? operands[2] : operands[1];
      rewriter.replaceOpWithNewOp<tpp::AddOp>(linalgOp, operands[0],
                                              operands[1], output);
      return success();
    }
    if (tpp::utils::isTppMatmul(linalgOp)) {
      rewriter.replaceOpWithNewOp<tpp::MatmulOp>(linalgOp, operands[0],
                                                 operands[1], operands[2]);
      return success();
    }
    return rewriter.notifyMatchFailure(
        linalgOp, "failed to match a known library_call attribute");
  }

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!linalgOp.hasBufferSemantics())
      return rewriter.notifyMatchFailure(linalgOp, "Expect buffer semantics");
    if (!tpp::utils::hasStaticShape(linalgOp))
      return rewriter.notifyMatchFailure(
          linalgOp, "Expect static shape when mapping to tpp");

    SmallVector<Value> newOperands;
    IRMapping mapping;
    for (Value operand : linalgOp->getOperands()) {
      if (failed(checkOperandForTpp(operand)))
        return rewriter.notifyMatchFailure(
            linalgOp, "Expect scalar or rank 2 memref when mapping to tpp");
      // This is not required anymore as we don't reshape anymore.
      // Will be clean-up when we introduce the matchers.
      newOperands.push_back(operand);
      mapping.map(operand, operand);
    }
    return rewriteToTppOp(linalgOp, newOperands, mapping, rewriter);
  }
};

// Convert a linalg.batch_reduce_matmul to a tpp.brgemm.
struct ConvertBrgemmToTpp
    : public OpRewritePattern<linalg::BatchReduceMatmulOp> {
  using OpRewritePattern<linalg::BatchReduceMatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::BatchReduceMatmulOp brMatmulOp,
                                PatternRewriter &rewriter) const override {
    if (!brMatmulOp.hasBufferSemantics())
      return rewriter.notifyMatchFailure(
          brMatmulOp, "Expect buffer semantics when mapping to tpp");
    if (!tpp::utils::hasStaticShape(brMatmulOp))
      return rewriter.notifyMatchFailure(
          brMatmulOp, "Expect static shape when mapping to tpp");
    SmallVector<Value> inputs = brMatmulOp.getDpsInputOperands();
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
    if (!matmulOp.hasBufferSemantics())
      return rewriter.notifyMatchFailure(
          matmulOp, "Expect buffer semantics when mapping to tpp");
    if (!tpp::utils::hasStaticShape(matmulOp))
      return rewriter.notifyMatchFailure(
          matmulOp, "Expect static shape when mapping to tpp");
    SmallVector<Value> inputs = matmulOp.getDpsInputOperands();
    SmallVector<Value> outputs = matmulOp.getDpsInitOperands();
    rewriter.replaceOpWithNewOp<tpp::MatmulOp>(matmulOp, inputs, outputs[0]);
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
               ConvertMatmulToTpp>(patterns.getContext());
  // clang-format on
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createConvertLinalgToTppPass() {
  return std::make_unique<ConvertLinalgToTpp>();
}
