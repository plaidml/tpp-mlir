//===- ConvertTppToXsmm.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/Tpp/TppOps.h"
#include "Standalone/Passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "Standalone/Passes.h.inc"

namespace {

struct ConvertTppMatmulOp : public OpRewritePattern<MatmulOp> {
  using OpRewritePattern<MatmulOp>::OpRewritePattern;

  SmallVector<Value, 4> getMemRefOperands(OpBuilder &b, Location loc,
                                          ValueRange operands) const {
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

  // TODO: take into account types
  std::string composeFunctionNameInvoke(MatmulOp matmulOp) const {
    return "xsmm_gemm_invoke";
  }

  std::string composeFunctionNameDispatch(MatmulOp matmulOp) const {
    return "xsmm_gemm_dispatch";
  }

  SmallVector<Type, 4>
  extractInvokeOperandTypes(SmallVector<Value, 6> operands) const {
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

  FlatSymbolRefAttr
  getLibraryCallSymbolRefInvoke(MatmulOp matmulOp,
                                SmallVector<Value, 6> operands,
                                PatternRewriter &rewriter) const {

    std::string fnName = composeFunctionNameInvoke(matmulOp);

    FlatSymbolRefAttr fnNameAttr =
        SymbolRefAttr::get(rewriter.getContext(), fnName);
    ModuleOp module = matmulOp->getParentOfType<ModuleOp>();
    if (module.lookupSymbol(fnNameAttr.getAttr()))
      return fnNameAttr;

    SmallVector<Type, 4> inputTypes(extractInvokeOperandTypes(operands));
    auto libFnType = rewriter.getFunctionType(inputTypes, {});
    OpBuilder::InsertionGuard guard(rewriter);
    // Insert before module terminator.
    rewriter.setInsertionPoint(module.getBody(),
                               std::prev(module.getBody()->end()));
    func::FuncOp funcOp = rewriter.create<func::FuncOp>(
        matmulOp->getLoc(), fnNameAttr.getValue(), libFnType);
    // Insert a function attribute that will trigger the emission of the
    // corresponding `_mlir_ciface_xxx` interface so that external libraries see
    // a normalized ABI. This interface is added during std to llvm conversion.
    funcOp->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                    UnitAttr::get(matmulOp->getContext()));
    funcOp.setPrivate();
    return fnNameAttr;
  }

  SmallVector<Type, 4> extractDispatchOperandTypes(MatmulOp matmulOp) const {
    Type integer32 = IntegerType::get(matmulOp.getContext(), 32);
    return {integer32 /*m*/,   integer32 /*n*/,   integer32 /*k*/,
            integer32 /*lda*/, integer32 /*ldb*/, integer32 /*ldc*/};
  }

  FlatSymbolRefAttr
  getLibraryCallSymbolRefDispatch(MatmulOp matmulOp,
                                  PatternRewriter &rewriter) const {
    std::string fnName = composeFunctionNameDispatch(matmulOp);

    FlatSymbolRefAttr fnNameAttr =
        SymbolRefAttr::get(rewriter.getContext(), fnName);
    ModuleOp module = matmulOp->getParentOfType<ModuleOp>();
    if (module.lookupSymbol(fnNameAttr.getAttr()))
      return fnNameAttr;

    SmallVector<Type, 4> inputTypes(extractDispatchOperandTypes(matmulOp));
    auto libFnType = rewriter.getFunctionType(
        inputTypes, IntegerType::get(rewriter.getContext(), 64));

    OpBuilder::InsertionGuard guard(rewriter);
    // Insert before module terminator.
    rewriter.setInsertionPoint(module.getBody(),
                               std::prev(module.getBody()->end()));
    func::FuncOp funcOp = rewriter.create<func::FuncOp>(
        matmulOp->getLoc(), fnNameAttr.getValue(), libFnType);
    // Insert a function attribute that will trigger the emission of the
    // corresponding `_mlir_ciface_xxx` interface so that external libraries see
    // a normalized ABI. This interface is added during std to llvm conversion.
    funcOp->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                    UnitAttr::get(matmulOp->getContext()));
    funcOp.setPrivate();
    return fnNameAttr;
  }

  Attribute getIntAttr(Builder &builder, IntegerType tp, int64_t val) const {
    return builder.getIntegerAttr(tp, APInt(tp.getWidth(), val));
  }

  SmallVector<Value, 6> getDispatchOperands(MatmulOp matmulOp,
                                            PatternRewriter &rewriter) const {
    MemRefType memrefC = matmulOp.getMatrixCType();
    MemRefType memrefA = matmulOp.getMatrixAType();
    int64_t m = memrefC.getShape()[0];
    int64_t n = memrefC.getShape()[1];
    int64_t k = memrefA.getShape()[1];
    int64_t lda = m;
    int64_t ldb = k;
    int64_t ldc = m;
    Location loc = matmulOp.getLoc();
    SmallVector<Value, 6> operands;

    IntegerType integer = IntegerType::get(rewriter.getContext(), 32);
    operands.push_back(rewriter.create<arith::ConstantOp>(
        loc, integer, getIntAttr(rewriter, integer, m)));
    operands.push_back(rewriter.create<arith::ConstantOp>(
        loc, integer, getIntAttr(rewriter, integer, n)));
    operands.push_back(rewriter.create<arith::ConstantOp>(
        loc, integer, getIntAttr(rewriter, integer, k)));
    operands.push_back(rewriter.create<arith::ConstantOp>(
        loc, integer, getIntAttr(rewriter, integer, lda)));
    operands.push_back(rewriter.create<arith::ConstantOp>(
        loc, integer, getIntAttr(rewriter, integer, ldb)));
    operands.push_back(rewriter.create<arith::ConstantOp>(
        loc, integer, getIntAttr(rewriter, integer, ldc)));
    return operands;
  }

  LogicalResult matchAndRewrite(MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    // build dispatch.
    auto libraryCallNameDispatch =
        getLibraryCallSymbolRefDispatch(matmulOp, rewriter);
    if (!libraryCallNameDispatch)
      return failure();
    Value dispatched =
        rewriter
            .create<func::CallOp>(
                matmulOp.getLoc(), libraryCallNameDispatch.getValue(),
                TypeRange{IntegerType::get(rewriter.getContext(), 64)},
                getDispatchOperands(matmulOp, rewriter))
            .getResult(0);

    SmallVector<Value, 6> invokeOperands;
    invokeOperands.push_back(dispatched);
    invokeOperands.append(matmulOp->getOperands().begin(),
                          matmulOp->getOperands().end());

    // build invoke.
    auto libraryCallNameInvoke =
        getLibraryCallSymbolRefInvoke(matmulOp, invokeOperands, rewriter);
    if (!libraryCallNameInvoke)
      return failure();
    rewriter.replaceOpWithNewOp<func::CallOp>(
        matmulOp, libraryCallNameInvoke.getValue(), TypeRange(),
        getMemRefOperands(rewriter, matmulOp.getLoc(), invokeOperands));
    return success();
  }
};

void populateTppToXsmmPatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<ConvertTppMatmulOp>(patterns.getContext());
  // clang-format on
}

// TODO: remove code duplication
// TODO: make xsmm a dialect (i.e., have an op to init the library).
struct ConvertTppToXsmm : public ConvertTppToXsmmBase<ConvertTppToXsmm> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateTppToXsmmPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::tpp::createConvertTppToXsmmPass() {
  return std::make_unique<ConvertTppToXsmm>();
}
