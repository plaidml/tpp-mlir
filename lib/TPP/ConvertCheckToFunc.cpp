//===- ConvertCheckToFunc.cpp -------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Check/CheckOps.h"
#include "TPP/Passes.h"
#include "TPP/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::check;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {
static SmallVector<Type> extractOperandTypes(OperandRange operands) {
  SmallVector<Type> results;
  results.reserve(operands.size());

  for (Value operand : operands) {
    Type operandType = operand.getType();
    if (auto memrefType = operandType.dyn_cast<MemRefType>()) {
      UnrankedMemRefType unrankedMemref = UnrankedMemRefType::get(
          memrefType.getElementType(), memrefType.getMemorySpace());
      results.push_back(unrankedMemref);
    } else
      results.push_back(operandType);
  }
  return results;
}

static SmallVector<Value> getMemRefOperands(OpBuilder &b, Location loc,
                                            ValueRange operands) {
  SmallVector<Value> res;
  res.reserve(operands.size());
  for (Value op : operands) {
    auto memrefType = op.getType().dyn_cast<MemRefType>();
    if (memrefType) {
      MemRefType rankedMemref = op.getType().dyn_cast<MemRefType>();
      UnrankedMemRefType unrankedMemref = UnrankedMemRefType::get(
          rankedMemref.getElementType(), rankedMemref.getMemorySpace());
      Value cast = b.create<memref::CastOp>(loc, unrankedMemref, op);
      res.push_back(cast);
    } else {
      res.push_back(op);
    }
  }
  return res;
}

static func::CallOp buildExpectTrueCall(Location loc, Operation *op,
                                        ModuleOp module,
                                        FlatSymbolRefAttr fnName,
                                        PatternRewriter &rewriter) {
  auto libFnType =
      rewriter.getFunctionType(extractOperandTypes(op->getOperands()), None);

  if (!module.lookupSymbol(fnName)) {
    OpBuilder::InsertionGuard guard(rewriter);
    // Insert before module terminator.
    rewriter.setInsertionPoint(module.getBody(),
                               std::prev(module.getBody()->end()));
    func::FuncOp funcOp =
        rewriter.create<func::FuncOp>(loc, fnName.getValue(), libFnType);

    funcOp->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                    UnitAttr::get(rewriter.getContext()));

    funcOp.setPrivate();
  }

  auto call = rewriter.create<func::CallOp>(
      loc, fnName.getValue(), TypeRange(),
      getMemRefOperands(rewriter, loc, op->getOperands()));
  return call;
}


static func::CallOp buildExpectAlmostEqualsCall(Location loc, Operation *op,
                                                ModuleOp module,
                                                FlatSymbolRefAttr fnName,
                                                PatternRewriter &rewriter) {
  auto libFnType =
      rewriter.getFunctionType(extractOperandTypes(op->getOperands()), None);

  if (!module.lookupSymbol(fnName.getAttr())) {
    OpBuilder::InsertionGuard guard(rewriter);
    // Insert before module terminator.
    rewriter.setInsertionPoint(module.getBody(),
                               std::prev(module.getBody()->end()));
    func::FuncOp funcOp =
        rewriter.create<func::FuncOp>(loc, fnName.getValue(), libFnType);
    funcOp->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                    UnitAttr::get(rewriter.getContext()));
    funcOp.setPrivate();
  }

  func::CallOp call = rewriter.create<func::CallOp>(
      loc, fnName.getValue(), TypeRange(),
      getMemRefOperands(rewriter, loc, op->getOperands()));
  return call;
}

struct ConvertExpectTrue : public OpRewritePattern<ExpectTrueOp> {
  ConvertExpectTrue(MLIRContext *context)
      : OpRewritePattern<ExpectTrueOp>(context, 1) {}

  LogicalResult matchAndRewrite(ExpectTrueOp expectTrueOp,
                                PatternRewriter &rewriter) const override {
    Location loc = expectTrueOp.getLoc();
    std::string kindAsString = "expect_true";

    FlatSymbolRefAttr fnName =
        SymbolRefAttr::get(rewriter.getContext(), kindAsString);

    ModuleOp module = expectTrueOp->getParentOfType<ModuleOp>();

    buildExpectTrueCall(loc, expectTrueOp, module, fnName, rewriter);
    rewriter.eraseOp(expectTrueOp);
    return success();
  }
};

struct ConvertAlmostEquals : public OpRewritePattern<ExpectAlmostEqOp> {
  ConvertAlmostEquals(MLIRContext *context)
      : OpRewritePattern<ExpectAlmostEqOp>(context, 1) {}

  LogicalResult matchAndRewrite(ExpectAlmostEqOp almostEqualsOp,
                                PatternRewriter &rewriter) const override {
    Location loc = almostEqualsOp.getLoc();
    std::string kindAsString = "expect_almost_equals";

    FlatSymbolRefAttr fnName =
        SymbolRefAttr::get(rewriter.getContext(), kindAsString);

    ModuleOp module = almostEqualsOp->getParentOfType<ModuleOp>();

    buildExpectAlmostEqualsCall(loc, almostEqualsOp, module, fnName, rewriter);
    rewriter.eraseOp(almostEqualsOp);
    return success();
  }
};

struct ConvertCheckToFunc : public ConvertCheckToFuncBase<ConvertCheckToFunc> {
  ConvertCheckToFunc() = default;
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    mlir::tpp::populateCheckToFuncPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // namespace

void mlir::tpp::populateCheckToFuncPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertAlmostEquals, ConvertExpectTrue>(patterns.getContext());
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::tpp::createConvertCheckToFuncPass() {
  return std::make_unique<ConvertCheckToFunc>();
}
