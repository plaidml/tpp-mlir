//===- ConvertXsmmToFunc.cpp -------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/Xsmm/XsmmOps.h"
#include "Standalone/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::xsmm;

#define GEN_PASS_CLASSES
#include "Standalone/Passes.h.inc"

namespace {

// TODO: rename TernaryCallOp -> TernaryOp
struct ConvertTernaryXsmmCallOp : public OpRewritePattern<TernaryCallOp> {
  using OpRewritePattern<TernaryCallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TernaryCallOp callOp,
                                PatternRewriter &rewriter) const override {
    return failure();
  }
};

struct ConvertDispatchCallOp : public OpRewritePattern<DispatchOp> {
  using OpRewritePattern<DispatchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DispatchOp dispatchOp,
                                PatternRewriter &rewriter) const override {
    Location loc = dispatchOp.getLoc();
    FlatSymbolRefAttr fnName =
        SymbolRefAttr::get(rewriter.getContext(), dispatchOp.getCallee());
    ModuleOp module = dispatchOp->getParentOfType<ModuleOp>();
    if (module.lookupSymbol(fnName.getAttr()))
      return failure();

    auto libFnType =
        rewriter.getFunctionType(dispatchOp.getOperandTypes(),
                                 IntegerType::get(rewriter.getContext(), 64));
    OpBuilder::InsertionGuard guard(rewriter);
    // Insert before module terminator.
    rewriter.setInsertionPoint(module.getBody(),
                               std::prev(module.getBody()->end()));
    func::FuncOp funcOp =
        rewriter.create<func::FuncOp>(loc, fnName.getValue(), libFnType);
    // Insert a function attribute that will trigger the emission of the
    // corresponding `_mlir_ciface_xxx` interface so that external libraries see
    // a normalized ABI.
    funcOp->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                    UnitAttr::get(dispatchOp->getContext()));
    funcOp.setPrivate();
    rewriter.replaceOp(dispatchOp, funcOp->getResult(0));
    return success();
  }
};

void populateXsmmToFuncPatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<ConvertTernaryXsmmCallOp,
               ConvertDispatchCallOp>(patterns.getContext());
  // clang-format on
}

struct ConvertXsmmToFunc : public ConvertXsmmToFuncBase<ConvertXsmmToFunc> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateXsmmToFuncPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::tpp::createConvertXsmmToFuncPass() {
  return std::make_unique<ConvertXsmmToFunc>();
}
