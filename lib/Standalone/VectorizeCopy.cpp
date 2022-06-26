//===- VectorizeCopy.cpp -----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::memref;

#define GEN_PASS_CLASSES
#include "Standalone/Passes.h.inc"

namespace {

struct VectorizeCopyOpWithLinalg : public OpRewritePattern<CopyOp> {
  using OpRewritePattern<CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CopyOp copy,
                                PatternRewriter &rewriter) const override {
    return linalg::vectorizeCopy(rewriter, copy);
  }
};

void populateCopyVectorizationPatterns(RewritePatternSet &patterns) {
  patterns.add<VectorizeCopyOpWithLinalg>(patterns.getContext());
}

// TODO: Can also choose to use a tpp.identity
struct VectorizeCopy : public VectorizeCopyBase<VectorizeCopy> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateCopyVectorizationPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createVectorizeCopyPass() {
  return std::make_unique<VectorizeCopy>();
}
