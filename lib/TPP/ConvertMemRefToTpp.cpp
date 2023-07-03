//===- LinalgMemRefToTpp.cpp -------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Tpp/TppOps.h"
#include "TPP/Dialect/Tpp/TppTraits.h"
#include "TPP/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

// Convert a memref.copy to a tpp.identity.
struct ConvertMemRefCopyToTpp : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    auto source = copyOp.getSource();
    auto sourceType = source.getType().cast<MemRefType>();
    if (sourceType.getRank() != 2 ||
        failed(OpTrait::tpp::verifyUnitStrideInnerLoop(
            copyOp, /*emitDiagnostic=*/false))) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<tpp::IdentityOp>(copyOp, copyOp.getSource(),
                                                 copyOp.getTarget());
    return success();
  }
};

struct ConvertMemRefToTpp : public ConvertMemRefToTppBase<ConvertMemRefToTpp> {
  ConvertMemRefToTpp() = default;
  void runOnOperation() override {
    MLIRContext *ctx = getOperation().getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<ConvertMemRefCopyToTpp>(ctx);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createConvertMemRefToTppPass() {
  return std::make_unique<ConvertMemRefToTpp>();
}
