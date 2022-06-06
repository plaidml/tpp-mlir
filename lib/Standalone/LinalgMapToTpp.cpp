//===- LinalgMapToTpp.cpp ----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/TppOps.h"
#include "Standalone/TppPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "Standalone/TppPasses.h.inc"

#define DEBUG_TYPE "linalg-map-to-tpp"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

namespace {

struct MapGenericOpToTpp : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    linalgOp->dump();
    return failure();
  }
};

void populateLinalgToTppPatterns(RewritePatternSet &patterns) {
  patterns.add<MapGenericOpToTpp>(patterns.getContext());
}

struct MapToTpp : public LinalgMapToTppBase<MapToTpp> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateLinalgToTppPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // end namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createMapLinalgToTppPass() {
  return std::make_unique<MapToTpp>();
}
