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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::xsmm;

#define GEN_PASS_CLASSES
#include "Standalone/Passes.h.inc"

namespace {

static SmallVector<Type, 4> extractInvokeOperandTypes(OperandRange operands) {
  SmallVector<Type, 4> result;
  result.reserve(operands.size());

  for (Value operand : operands) {
    Type operandType = operand.getType();
    if (auto memrefType = operandType.dyn_cast<MemRefType>()) {
      UnrankedMemRefType unrankedMemref = UnrankedMemRefType::get(
          memrefType.getElementType(), memrefType.getMemorySpace());
      result.push_back(unrankedMemref);
    } else
      result.push_back(operandType);
  }
  return result;
}

static SmallVector<Value, 4> getMemRefOperands(OpBuilder &b, Location loc,
                                               ValueRange operands) {
  SmallVector<Value, 4> res;
  res.reserve(operands.size());
  for (auto op : operands) {
    auto memrefType = op.getType().dyn_cast<MemRefType>();
    if (!memrefType) {
      res.push_back(op);
      continue;
    }
    MemRefType rankedMemref = op.getType().dyn_cast<MemRefType>();
    UnrankedMemRefType unrankedMemref = UnrankedMemRefType::get(
        rankedMemref.getElementType(), rankedMemref.getMemorySpace());
    Value cast = b.create<memref::CastOp>(loc, unrankedMemref, op);
    res.push_back(cast);
  }
  return res;
}

// TODO: rename TernaryCallOp -> TernaryOp
// TODO: Interface for TernaryCallOp, BinaryCallOp and UnaryCallOp.
// to avoid code duplication.
struct ConvertTernaryXsmmCallOp : public OpRewritePattern<TernaryCallOp> {
  using OpRewritePattern<TernaryCallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TernaryCallOp callOp,
                                PatternRewriter &rewriter) const override {
    Location loc = callOp.getLoc();
    FlatSymbolRefAttr fnName =
        SymbolRefAttr::get(rewriter.getContext(), callOp.getCallee());
    ModuleOp module = callOp->getParentOfType<ModuleOp>();
    if (module.lookupSymbol(fnName.getAttr()))
      return failure();

    auto libFnType = rewriter.getFunctionType(
        extractInvokeOperandTypes(callOp.getOperands()), {});
    {
      OpBuilder::InsertionGuard guard(rewriter);
      // Insert before module terminator.
      rewriter.setInsertionPoint(module.getBody(),
                                 std::prev(module.getBody()->end()));
      func::FuncOp funcOp =
          rewriter.create<func::FuncOp>(loc, fnName.getValue(), libFnType);
      // Insert a function attribute that will trigger the emission of the
      // corresponding `_mlir_ciface_xxx` interface so that external libraries
      // see a normalized ABI.
      funcOp->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                      UnitAttr::get(callOp->getContext()));
      funcOp.setPrivate();
    }

    rewriter.create<func::CallOp>(
        loc, fnName.getValue(), TypeRange(),
        getMemRefOperands(rewriter, loc, callOp.getOperands()));
    rewriter.eraseOp(callOp);
    return success();
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
    {
      OpBuilder::InsertionGuard guard(rewriter);
      // Insert before module terminator.
      rewriter.setInsertionPoint(module.getBody(),
                                 std::prev(module.getBody()->end()));
      func::FuncOp funcOp =
          rewriter.create<func::FuncOp>(loc, fnName.getValue(), libFnType);
      // Insert a function attribute that will trigger the emission of the
      // corresponding `_mlir_ciface_xxx` interface so that external libraries
      // see a normalized ABI.
      funcOp->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                      UnitAttr::get(dispatchOp->getContext()));
      funcOp.setPrivate();
    }

    rewriter.replaceOpWithNewOp<func::CallOp>(
        dispatchOp, fnName.getValue(),
        IntegerType::get(rewriter.getContext(), 64), dispatchOp.getOperands());
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
