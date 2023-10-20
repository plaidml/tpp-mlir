//===- ConvertMemRefToXsmm.cpp -----------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Xsmm/XsmmOps.h"
#include "TPP/Dialect/Xsmm/XsmmUtils.h"
#include "TPP/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

// Convert a memref.copy to a xsmm identity.
struct ConvertMemRefCopyToXsmm : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    Value source = copyOp.getSource();
    Value dest = copyOp.getTarget();
    auto unaryInfo =
        xsmm::utils::getUnaryInfo(source, dest, xsmm::UnaryFlags::NONE);
    if (failed(unaryInfo))
      return failure();
    auto flags = rewriter.getArrayAttr(xsmm::UnaryFlagsAttr::get(
        rewriter.getContext(), xsmm::UnaryFlags::NONE));
    xsmm::UnaryKindAttr kind = xsmm::UnaryKindAttr::get(
        rewriter.getContext(), xsmm::UnaryKind::IDENTITY);
    SmallVector<Value> operands{source, dest};
    xsmm::utils::replaceOpWithUnary(rewriter, copyOp, operands, *unaryInfo,
                                    flags, kind);
    return success();
  }
};

struct ConvertMemRefToXsmm
    : public ConvertMemRefToXsmmBase<ConvertMemRefToXsmm> {
  ConvertMemRefToXsmm() = default;
  void runOnOperation() override {
    MLIRContext *ctx = getOperation().getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<ConvertMemRefCopyToXsmm>(ctx);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createConvertMemRefToXsmmPass() {
  return std::make_unique<ConvertMemRefToXsmm>();
}
