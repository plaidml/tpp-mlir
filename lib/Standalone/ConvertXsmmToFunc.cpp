//===- ConvertXsmmToFunc.cpp -------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/Xsmm/XsmmAttr.h"
#include "Standalone/Dialect/Xsmm/XsmmOps.h"
#include "Standalone/Passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::xsmm;

#define GEN_PASS_CLASSES
#include "Standalone/Passes.h.inc"

namespace {

// Cast memref to unranked memref and leave all the other operands as they are.
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

// Similar to 'extractInvokeOperandTypes' but acting on Value. Memref
// are casted by introducing castOp. We cast the memref to clear the shape
// and have a single function signature in the runtime.
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

static LogicalResult buildInvokeCall(Location loc, std::string funcName,
                                     Operation *op, PatternRewriter &rewriter) {
  funcName = "xsmm_" + funcName + "_invoke";
  FlatSymbolRefAttr fnName = SymbolRefAttr::get(op->getContext(), funcName);

  ModuleOp module = op->getParentOfType<ModuleOp>();
  auto libFnType = rewriter.getFunctionType(
      extractInvokeOperandTypes(op->getOperands()), {});

  if (!module.lookupSymbol(fnName)) {
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
                    UnitAttr::get(op->getContext()));
    funcOp.setPrivate();
  }

  rewriter.create<func::CallOp>(
      loc, fnName.getValue(), TypeRange(),
      getMemRefOperands(rewriter, loc, op->getOperands()));
  return success();
}

struct ConvertTernaryXsmmOp : public OpRewritePattern<TernaryOp> {
  using OpRewritePattern<TernaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TernaryOp ternaryOp,
                                PatternRewriter &rewriter) const override {
    if (succeeded(buildInvokeCall(ternaryOp.getLoc(),
                                  stringifyEnum(ternaryOp.getCallee()).str(),
                                  ternaryOp, rewriter))) {
      rewriter.eraseOp(ternaryOp);
      return success();
    }
    return failure();
  }
};

struct ConvertUnaryXsmmOp : public OpRewritePattern<UnaryOp> {
  using OpRewritePattern<UnaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(UnaryOp unaryOp,
                                PatternRewriter &rewriter) const override {
    // Handle the scalar case. There is no operator overloading
    // in MLIR (thus we need to change the function name from
    // "unary" to "unary_scalar"). We also don't want to convert
    // the scalar to a memref by using an alloc/alloca.
    std::string funcName = "unary";
    if (unaryOp.hasScalarInput())
      funcName = "unary_scalar";
    if (succeeded(
            buildInvokeCall(unaryOp.getLoc(), funcName, unaryOp, rewriter))) {
      rewriter.eraseOp(unaryOp);
      return success();
    }
    return failure();
  }
};

struct ConvertBinaryXsmmOp : public OpRewritePattern<BinaryOp> {
  using OpRewritePattern<BinaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BinaryOp binaryOp,
                                PatternRewriter &rewriter) const override {
    if (succeeded(buildInvokeCall(binaryOp.getLoc(),
                                  stringifyEnum(binaryOp.getCallee()).str(),
                                  binaryOp, rewriter))) {
      rewriter.eraseOp(binaryOp);
      return success();
    }
    return failure();
  }
};

static func::CallOp buildDispatchCall(Location loc,
                                      ArrayRef<Value> dispatchOperands,
                                      ArrayRef<Type> dispatchOperandTypes,
                                      ModuleOp module, FlatSymbolRefAttr fnName,
                                      PatternRewriter &rewriter) {
  auto libFnType = rewriter.getFunctionType(
      dispatchOperandTypes, IntegerType::get(rewriter.getContext(), 64));

  if (!module.lookupSymbol(fnName.getAttr())) {
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
                    UnitAttr::get(rewriter.getContext()));
    funcOp.setPrivate();
  }

  func::CallOp call = rewriter.create<func::CallOp>(
      loc, fnName.getValue(), IntegerType::get(rewriter.getContext(), 64),
      dispatchOperands);
  return call;
}

struct ConvertTernaryDispatch : public OpRewritePattern<TernaryDispatchOp> {
  using OpRewritePattern<TernaryDispatchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TernaryDispatchOp dispatchOp,
                                PatternRewriter &rewriter) const override {
    Location loc = dispatchOp.getLoc();
    std::string kindAsString = stringifyEnum(dispatchOp.getKind()).str();
    kindAsString = "xsmm_" + kindAsString + "_dispatch";
    FlatSymbolRefAttr fnName =
        SymbolRefAttr::get(rewriter.getContext(), kindAsString);

    ModuleOp module = dispatchOp->getParentOfType<ModuleOp>();
    SmallVector<Value, 10> dispatchOperands;
    SmallVector<Type, 10> dispatchOperandTypes;
    IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
    ArrayRef<int64_t> integers = dispatchOp.getInputsAttr().asArrayRef();
    size_t arrayAttrSize = integers.size();
    for (size_t idx = 0; idx < arrayAttrSize; idx++) {
      IntegerAttr attr = IntegerAttr::get(rewriter.getI64Type(), integers[idx]);
      dispatchOperands.push_back(
          rewriter.create<arith::ConstantOp>(loc, integer64, attr));
      dispatchOperandTypes.push_back(integer64);
    }
    func::CallOp call = buildDispatchCall(
        loc, dispatchOperands, dispatchOperandTypes, module, fnName, rewriter);
    rewriter.replaceOp(dispatchOp, call.getResult(0));
    return success();
  }
};

struct ConvertBinaryDispatch : public OpRewritePattern<BinaryDispatchOp> {
  using OpRewritePattern<BinaryDispatchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BinaryDispatchOp dispatchOp,
                                PatternRewriter &rewriter) const override {
    return failure();
  }
};

struct ConvertUnaryDispatch : public OpRewritePattern<UnaryDispatchOp> {
  using OpRewritePattern<UnaryDispatchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(UnaryDispatchOp dispatchOp,
                                PatternRewriter &rewriter) const override {
    Location loc = dispatchOp.getLoc();
    std::string kindAsString = "xsmm_unary_dispatch";
    FlatSymbolRefAttr fnName =
        SymbolRefAttr::get(rewriter.getContext(), kindAsString);

    ModuleOp module = dispatchOp->getParentOfType<ModuleOp>();
    SmallVector<Value, 10> dispatchOperands;
    SmallVector<Type, 10> dispatchOperandTypes;
    IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
    ArrayRef<int64_t> integers = dispatchOp.getInputsAttr().asArrayRef();
    size_t arrayAttrSize = integers.size();
    for (size_t idx = 0; idx < arrayAttrSize; idx++) {
      IntegerAttr attr = IntegerAttr::get(rewriter.getI64Type(), integers[idx]);
      dispatchOperands.push_back(
          rewriter.create<arith::ConstantOp>(loc, integer64, attr));
      dispatchOperandTypes.push_back(integer64);
    }

    // kind of operation to invoke.
    dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
        loc, integer64, dispatchOp.getKindAttr()));
    dispatchOperandTypes.push_back(integer64);

    // kind of broadcast
    dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
        loc, integer64, dispatchOp.getFlagsAttr()));
    dispatchOperandTypes.push_back(integer64);

    func::CallOp call = buildDispatchCall(
        loc, dispatchOperands, dispatchOperandTypes, module, fnName, rewriter);
    rewriter.replaceOp(dispatchOp, call.getResult(0));
    return success();
  }
};

void populateXsmmToFuncPatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<ConvertTernaryXsmmOp,
               ConvertBinaryXsmmOp,
               ConvertUnaryXsmmOp,
               ConvertTernaryDispatch,
               ConvertBinaryDispatch,
               ConvertUnaryDispatch>(patterns.getContext());
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
