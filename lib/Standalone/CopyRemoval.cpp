//===- CopyRemoval.cpp -------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/TppPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::memref;

#define GEN_PASS_CLASSES
#include "Standalone/TppPasses.h.inc"

namespace {

struct RemoveTrivialCopies : public OpRewritePattern<CopyOp> {
  using OpRewritePattern<CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CopyOp copy,
                                PatternRewriter &rewriter) const override {
    MemRefType sourceType = copy.source().getType().cast<MemRefType>();
    MemRefType targetType = copy.target().getType().cast<MemRefType>();
    if (sourceType != targetType)
      return failure();
    copy.target().replaceAllUsesWith(copy.source());
    rewriter.eraseOp(copy);
    return success();
  }
};

void populateCopyRemovalPatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<RemoveTrivialCopies>(patterns.getContext());
  // clang-format on
}

struct CopyRemoval : public CopyRemovalBase<CopyRemoval> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateCopyRemovalPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createCopyRemovalPass() {
  return std::make_unique<CopyRemoval>();
}
