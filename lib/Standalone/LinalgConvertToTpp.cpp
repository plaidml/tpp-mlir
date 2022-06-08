//===- LinalgConvertToTpp.cpp ------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/TppOps.h"
#include "Standalone/TppPasses.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "Standalone/TppPasses.h.inc"

#define DEBUG_TYPE "linalg-convert-to-tpp"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

namespace {

struct ConvertGenericOpToTpp : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!linalgOp.hasBufferSemantics())
      return failure();
    std::string libraryCall = linalgOp.getLibraryCallName();
    // TODO: find better way to express this.
    if (libraryCall.compare("tpp.identity") == 0) {
      rewriter.replaceOpWithNewOp<tpp::IdentityOp>(
          linalgOp, linalgOp->getOperand(0), linalgOp->getOperand(1));
      return success();
    }
    if (libraryCall.compare("tpp.relu") == 0) {
      rewriter.replaceOpWithNewOp<tpp::ReluOp>(
          linalgOp, linalgOp->getOperand(0), linalgOp->getOperand(1));
      return success();
    }
    if (libraryCall.compare("tpp.add") == 0) {
      rewriter.replaceOpWithNewOp<tpp::AddOp>(linalgOp, linalgOp->getOperand(0),
                                              linalgOp->getOperand(1),
                                              linalgOp->getOperand(2));
      return success();
    }
    if (libraryCall.compare("tpp.matmul") == 0) {
      rewriter.replaceOpWithNewOp<tpp::MatmulOp>(
          linalgOp, linalgOp->getOperand(0), linalgOp->getOperand(1),
          linalgOp->getOperand(2));
      return success();
    }
    return failure();
  }
};

void populateLinalgToTppPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertGenericOpToTpp>(patterns.getContext());
}

struct ConvertLinalgToTpp : public ConvertLinalgToTppBase<ConvertLinalgToTpp> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateLinalgToTppPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // end namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createConvertLinalgToTppPass() {
  return std::make_unique<ConvertLinalgToTpp>();
}
