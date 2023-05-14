//===- CombineTpp.cpp --------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Tpp/TppOps.h"
#include "TPP/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

struct CombineBrgemmAddAndRelu : public OpRewritePattern<tpp::ReluOp> {
  using OpRewritePattern<tpp::ReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpp::ReluOp reluOp,
                                PatternRewriter &rewriter) const override {
    if (!reluOp.hasTensorSemantics())
      return failure();
    Value operandRelu = reluOp.getInputs()[0];
    auto maybeAdd = operandRelu.getDefiningOp<tpp::AddOp>();
    if (!maybeAdd)
      return failure();
    SmallVector<Value> brgemmOperands;
    Value addOperand;
    bool hasBrgemmProducer = false;
    for (Value operand : maybeAdd.getInputs()) {
      if (auto brgemmOp = operand.getDefiningOp<tpp::BrgemmOp>()) {
        brgemmOperands = brgemmOp.getInputs();
        hasBrgemmProducer = true;
        continue;
      }
      addOperand = operand;
    }
    if (!hasBrgemmProducer)
      return failure();
    brgemmOperands.push_back(addOperand);
    auto ctx = rewriter.getContext();
    auto unaryType =
        tpp::FusedUnaryOpTypeAttr::get(ctx, tpp::FusedUnaryOpType::RELU);
    auto binaryType =
        tpp::FusedBinaryOpTypeAttr::get(ctx, tpp::FusedBinaryOpType::ADD);
    rewriter.replaceOpWithNewOp<tpp::FusedBrgemmOp>(
        reluOp, brgemmOperands, addOperand, unaryType, binaryType);
    return success();
  }
};

void populatePatterns(RewritePatternSet &patterns) {
  patterns.add<CombineBrgemmAddAndRelu>(patterns.getContext());
}

struct CombineTppOps : public CombineTppOpsBase<CombineTppOps> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populatePatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::tpp::createCombineTppPass() {
  return std::make_unique<CombineTppOps>();
}
