//===- ConvertXsmmToFunc.cpp -------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Xsmm/XsmmEnum.h"
#include "TPP/Dialect/Xsmm/XsmmOps.h"
#include "TPP/Passes.h"
#include "TPP/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::xsmm;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

// Cast memref to unranked memref and leave all the other operands as they are.
static SmallVector<Type> extractInvokeOperandTypes(OperandRange operands,
                                                   PatternRewriter &rewriter) {
  SmallVector<Type> results;
  // One extra operand for datatype
  results.reserve(operands.size() + 1);
  IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
  results.push_back(integer64);
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

static SmallVector<Type>
extractInvokeOperandTypesForMeta(OperandRange operands, IndexType indexType,
                                 PatternRewriter &rewriter) {
  SmallVector<Type> results;
  // One extra operand for datatype
  IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
  results.push_back(integer64);
  for (Value operand : operands) {
    Type operandType = operand.getType();
    if (auto memrefType = operandType.dyn_cast<MemRefType>()) {
      // TODO: non-POD will require an LLVMTypeConverter.
      Type basePtrType =
          LLVM::LLVMPointerType::get(memrefType.getElementType());
      results.push_back(basePtrType);
      results.push_back(indexType); // offset
    } else {
      results.push_back(operand.getType());
    }
  }
  return results;
}

// Similar to 'extractInvokeOperandTypes' but acting on Value. Memref
// are casted by introducing castOp. We cast the memref to clear the shape
// and have a single function signature in the runtime.
static SmallVector<Value> getMemRefOperands(OpBuilder &b, Location loc,
                                            ValueRange operands,
                                            IntegerAttr dataTypeAttr) {
  SmallVector<Value> res;
  // One extra operand for datatype
  res.reserve(operands.size() + 1);
  IntegerType integer64 = IntegerType::get(b.getContext(), 64);
  res.push_back(b.create<arith::ConstantOp>(loc, integer64, dataTypeAttr));
  llvm::DenseMap<Value, Value> operandToCastedOperand;
  for (Value operand : operands) {
    auto memrefType = operand.getType().dyn_cast<MemRefType>();
    if (!memrefType)
      res.push_back(operand);
    else {
      MemRefType rankedMemref = operand.getType().dyn_cast<MemRefType>();
      UnrankedMemRefType unrankedMemref = UnrankedMemRefType::get(
          rankedMemref.getElementType(), rankedMemref.getMemorySpace());
      if (operandToCastedOperand.count(operand)) {
        res.push_back(operandToCastedOperand[operand]);
        continue;
      }
      Value cast = b.create<memref::CastOp>(loc, unrankedMemref, operand);
      operandToCastedOperand[operand] = cast;
      res.push_back(cast);
    }
  }
  return res;
}

static SmallVector<Value>
getMemRefOperandsUsingMetadata(OpBuilder &builder, Location loc,
                               ValueRange operands, IntegerAttr dataTypeAttr) {
  SmallVector<Value> res;
  IntegerType integer64 = IntegerType::get(builder.getContext(), 64);
  res.push_back(
      builder.create<arith::ConstantOp>(loc, integer64, dataTypeAttr));

  for (Value operand : operands) {
    auto memrefType = operand.getType().dyn_cast<MemRefType>();
    if (!memrefType) {
      res.push_back(operand);
      continue;
    }
    MemRefType baseMemrefType =
        MemRefType::get({}, memrefType.getElementType());
    Type basePtrType = builder.getIndexType();
    Type offsetType = builder.getIndexType();
    SmallVector<Type> sizesTypes(memrefType.getRank(), offsetType);
    SmallVector<Type> stridesTypes(memrefType.getRank(), offsetType);
    auto meta = builder.create<memref::ExtractStridedMetadataOp>(
        loc, baseMemrefType, offsetType, sizesTypes, stridesTypes, operand);
    Value basePointerAsIndex =
        builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, basePtrType,
                                                               operand);
    Value basePointerAsI64 = builder.create<arith::IndexCastOp>(
        loc, builder.getIntegerType(64), basePointerAsIndex);

    // TODO: non-POD will require an LLVMTypeConverter.
    Value basePointer = builder.create<LLVM::IntToPtrOp>(
        loc, LLVM::LLVMPointerType::get(memrefType.getElementType()),
        basePointerAsI64);
    res.push_back(basePointer);
    res.push_back(meta.getOffset());
  }
  return res;
}

static LogicalResult buildInvokeCall(Location loc, std::string funcName,
                                     Operation *op, bool useMeta,
                                     PatternRewriter &rewriter,
                                     IntegerAttr dataTypeAttr) {
  FlatSymbolRefAttr fnName = SymbolRefAttr::get(op->getContext(), funcName);
  ModuleOp module = op->getParentOfType<ModuleOp>();
  auto libFnType = rewriter.getFunctionType(
      (!useMeta) ? extractInvokeOperandTypes(op->getOperands(), rewriter)
                 : extractInvokeOperandTypesForMeta(
                       op->getOperands(), rewriter.getIndexType(), rewriter),
      {});

  if (!module.lookupSymbol(fnName)) {
    OpBuilder::InsertionGuard guard(rewriter);
    // Insert before module terminator.
    rewriter.setInsertionPoint(module.getBody(),
                               std::prev(module.getBody()->end()));
    func::FuncOp funcOp =
        rewriter.create<func::FuncOp>(loc, fnName.getValue(), libFnType);
    if (!useMeta) {
      // Insert a function attribute that will trigger the emission of the
      // corresponding `_mlir_ciface_xxx` interface so that external libraries
      // see a normalized ABI.
      funcOp->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                      UnitAttr::get(op->getContext()));
    }
    funcOp.setPrivate();
  }

  rewriter.create<func::CallOp>(
      loc, fnName.getValue(), TypeRange(),
      (!useMeta)
          ? getMemRefOperands(rewriter, loc, op->getOperands(), dataTypeAttr)
          : getMemRefOperandsUsingMetadata(rewriter, loc, op->getOperands(),
                                           dataTypeAttr));
  return success();
}

struct ConvertTernaryXsmmOp : public OpRewritePattern<TernaryOp> {
  ConvertTernaryXsmmOp(MLIRContext *context, bool useMeta,
                       PatternBenefit benefit = 1)
      : OpRewritePattern<TernaryOp>(context, benefit), useMeta(useMeta) {}

  LogicalResult matchAndRewrite(TernaryOp ternaryOp,
                                PatternRewriter &rewriter) const override {
    std::string funcName =
        "xsmm_" + stringifyEnum(ternaryOp.getCallee()).str() + "_invoke";
    if (succeeded(buildInvokeCall(ternaryOp.getLoc(), funcName, ternaryOp,
                                  useMeta, rewriter,
                                  ternaryOp.getDataTypeAttr()))) {
      rewriter.eraseOp(ternaryOp);
      return success();
    }
    return failure();
  }

private:
  bool useMeta = false;
};

struct ConvertGemmXsmmOp : public OpRewritePattern<GemmOp> {
  ConvertGemmXsmmOp(MLIRContext *context, bool useMeta,
                    PatternBenefit benefit = 1)
      : OpRewritePattern<GemmOp>(context, benefit), useMeta(useMeta) {}

  LogicalResult matchAndRewrite(GemmOp gemmOp,
                                PatternRewriter &rewriter) const override {
    std::string funcName = "xsmm_gemm_invoke";
    if (succeeded(buildInvokeCall(gemmOp.getLoc(), funcName, gemmOp, useMeta,
                                  rewriter, gemmOp.getDataTypeAttr()))) {
      rewriter.eraseOp(gemmOp);
      return success();
    }
    return failure();
  }

private:
  bool useMeta = false;
};

struct ConvertBrgemmXsmmOp : public OpRewritePattern<BrgemmOp> {
  ConvertBrgemmXsmmOp(MLIRContext *context, bool useMeta,
                      PatternBenefit benefit = 1)
      : OpRewritePattern<BrgemmOp>(context, benefit), useMeta(useMeta) {}

  LogicalResult matchAndRewrite(BrgemmOp brgemmOp,
                                PatternRewriter &rewriter) const override {
    std::string funcName = "xsmm_brgemm_invoke";
    if (succeeded(buildInvokeCall(brgemmOp.getLoc(), funcName, brgemmOp,
                                  useMeta, rewriter,
                                  brgemmOp.getDataTypeAttr()))) {
      rewriter.eraseOp(brgemmOp);
      return success();
    }
    return failure();
  }

private:
  bool useMeta = false;
};

struct ConvertQuarternaryXsmmOp : public OpRewritePattern<QuarternaryOp> {
  ConvertQuarternaryXsmmOp(MLIRContext *context, bool useMeta,
                           PatternBenefit benefit = 1)
      : OpRewritePattern<QuarternaryOp>(context, benefit), useMeta(useMeta) {}

  LogicalResult matchAndRewrite(QuarternaryOp quarternaryOp,
                                PatternRewriter &rewriter) const override {
    std::string funcName =
        "xsmm_" + stringifyEnum(quarternaryOp.getCallee()).str() + "_invoke";
    if (succeeded(buildInvokeCall(quarternaryOp.getLoc(), funcName,
                                  quarternaryOp, useMeta, rewriter,
                                  quarternaryOp.getDataTypeAttr()))) {
      rewriter.eraseOp(quarternaryOp);
      return success();
    }
    return failure();
  }

private:
  bool useMeta = false;
};

struct ConvertUnaryXsmmOp : public OpRewritePattern<UnaryOp> {
  ConvertUnaryXsmmOp(MLIRContext *context, bool useMeta,
                     PatternBenefit benefit = 1)
      : OpRewritePattern<UnaryOp>(context, benefit), useMeta(useMeta) {}

  LogicalResult matchAndRewrite(UnaryOp unaryOp,
                                PatternRewriter &rewriter) const override {
    // Handle the scalar case. There is no operator overloading
    // in MLIR (thus we need to change the function name from
    // "unary" to "unary_scalar"). We also don't want to convert
    // the scalar to a memref by using an alloc/alloca.
    std::string funcName = "xsmm_unary_invoke";
    if (unaryOp.hasScalarInput())
      funcName = "xsmm_unary_scalar_invoke";
    if (succeeded(buildInvokeCall(unaryOp.getLoc(), funcName, unaryOp, useMeta,
                                  rewriter, unaryOp.getDataTypeAttr()))) {
      rewriter.eraseOp(unaryOp);
      return success();
    }
    return failure();
  }

private:
  bool useMeta = false;
};

struct ConvertBinaryXsmmOp : public OpRewritePattern<BinaryOp> {
  ConvertBinaryXsmmOp(MLIRContext *context, bool useMeta,
                      PatternBenefit benefit = 1)
      : OpRewritePattern<BinaryOp>(context, benefit), useMeta(useMeta) {}

  LogicalResult matchAndRewrite(BinaryOp binaryOp,
                                PatternRewriter &rewriter) const override {
    std::string funcName = "xsmm_binary_invoke";
    if (succeeded(buildInvokeCall(binaryOp.getLoc(), funcName, binaryOp,
                                  useMeta, rewriter,
                                  binaryOp.getDataTypeAttr()))) {
      rewriter.eraseOp(binaryOp);
      return success();
    }
    return failure();
  }

private:
  bool useMeta = false;
};

// TODO: move rewriter as first arg.
static func::CallOp buildDispatchCall(Location loc,
                                      ArrayRef<Value> dispatchOperands,
                                      ArrayRef<Type> dispatchOperandTypes,
                                      ModuleOp module, FlatSymbolRefAttr fnName,
                                      bool useMeta, RewriterBase &rewriter) {
  auto libFnType = rewriter.getFunctionType(
      dispatchOperandTypes, IntegerType::get(rewriter.getContext(), 64));

  if (!module.lookupSymbol(fnName.getAttr())) {
    OpBuilder::InsertionGuard guard(rewriter);
    // Insert before module terminator.
    rewriter.setInsertionPoint(module.getBody(),
                               std::prev(module.getBody()->end()));
    func::FuncOp funcOp =
        rewriter.create<func::FuncOp>(loc, fnName.getValue(), libFnType);
    if (!useMeta) {
      // Insert a function attribute that will trigger the emission of the
      // corresponding `_mlir_ciface_xxx` interface so that external libraries
      // see a normalized ABI.
      funcOp->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                      UnitAttr::get(rewriter.getContext()));
    }
    funcOp.setPrivate();
  }

  func::CallOp call = rewriter.create<func::CallOp>(
      loc, fnName.getValue(), IntegerType::get(rewriter.getContext(), 64),
      dispatchOperands);
  return call;
}

template <typename OpTy,
          typename = std::enable_if_t<
              std::is_same<OpTy, xsmm::UnaryDispatchOp>::value ||
              std::is_same<OpTy, xsmm::BinaryDispatchOp>::value ||
              std::is_same<OpTy, xsmm::TernaryDispatchOp>::value>>
void addKindOperand(RewriterBase &rewriter, OpTy dispatchOp,
                    SmallVectorImpl<Value> &dispatchOperands,
                    SmallVectorImpl<Type> &dispatchOperandTypes) {
  Location loc = dispatchOp.getLoc();
  IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
  dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
      loc, integer64, dispatchOp.getKindAttr()));
  dispatchOperandTypes.push_back(integer64);
}

void addKindOperand(RewriterBase &rewriter, GemmDispatchOp dispatchOp,
                    SmallVectorImpl<Value> &dispatchOperands,
                    SmallVectorImpl<Type> &dispatchOperandTypes) {
  /* do nothing */
}

void addKindOperand(RewriterBase &rewriter, BrgemmDispatchOp dispatchOp,
                    SmallVectorImpl<Value> &dispatchOperands,
                    SmallVectorImpl<Type> &dispatchOperandTypes) {
  /* do nothing */
}

template <typename OpTy>
static LogicalResult buildDispatchOp(RewriterBase &rewriter, OpTy dispatchOp,
                                     std::string funcName, bool useMeta) {
  Location loc = dispatchOp.getLoc();
  FlatSymbolRefAttr fnName =
      SymbolRefAttr::get(rewriter.getContext(), funcName);

  ModuleOp module = dispatchOp->template getParentOfType<ModuleOp>();
  SmallVector<Value, 10> dispatchOperands;
  SmallVector<Type, 10> dispatchOperandTypes;
  IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);

  // If `OpTy` is unary, binary or ternary we need to dispatch and extra
  // integer for the kind of operation to invoke.
  if (std::is_same<OpTy, xsmm::UnaryDispatchOp>::value ||
      std::is_same<OpTy, xsmm::BinaryDispatchOp>::value ||
      std::is_same<OpTy, xsmm::TernaryDispatchOp>::value) {
    addKindOperand(rewriter, dispatchOp, dispatchOperands,
                   dispatchOperandTypes);
  }

  // Dispatch the data type.
  dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
      loc, integer64, dispatchOp.getDataTypeAttr()));
  dispatchOperandTypes.push_back(integer64);

  // Dispatch the inputs.
  ArrayRef<int64_t> integers = dispatchOp.getInputsAttr().asArrayRef();
  size_t arrayAttrSize = integers.size();
  for (size_t idx = 0; idx < arrayAttrSize; idx++) {
    IntegerAttr attr = IntegerAttr::get(rewriter.getI64Type(), integers[idx]);
    dispatchOperands.push_back(
        rewriter.create<arith::ConstantOp>(loc, integer64, attr));
    dispatchOperandTypes.push_back(integer64);
  }

  // Dispatch the flags. Pass to the library the already ored-flag to
  // avoid changing the interface every time we add a new flag. Flags
  // are assumed to be verified before (i.e., op verifier).
  int64_t oredFlag = 0;
  for (auto flag : dispatchOp.getFlagsAttr()) {
    int64_t intAttr = flag.template dyn_cast<IntegerAttr>().getInt();
    // LIBXSMM is col-major, swap A and B flags.
    if (auto gemmFlag = dyn_cast_or_null<xsmm::GemmFlagsAttr>(flag)) {
      if (gemmFlag.getValue() == GemmFlags::VNNI_A)
        intAttr = static_cast<int64_t>(GemmFlags::VNNI_B);
      if (gemmFlag.getValue() == GemmFlags::VNNI_B)
        intAttr = static_cast<int64_t>(GemmFlags::VNNI_A);
    }
    oredFlag |= intAttr;
  }
  dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
      loc, integer64, IntegerAttr::get(rewriter.getI64Type(), oredFlag)));
  dispatchOperandTypes.push_back(integer64);

  func::CallOp call =
      buildDispatchCall(loc, dispatchOperands, dispatchOperandTypes, module,
                        fnName, useMeta, rewriter);
  rewriter.replaceOp(dispatchOp, call.getResult(0));
  return success();
}

struct ConvertGemmDispatchOp : public OpRewritePattern<GemmDispatchOp> {
  ConvertGemmDispatchOp(MLIRContext *context, bool useMeta,
                        PatternBenefit benefit = 1)
      : OpRewritePattern<GemmDispatchOp>(context, benefit), useMeta(useMeta) {}

  LogicalResult matchAndRewrite(GemmDispatchOp dispatchOp,
                                PatternRewriter &rewriter) const override {
    return buildDispatchOp<GemmDispatchOp>(rewriter, dispatchOp,
                                           "xsmm_matmul_dispatch", useMeta);
  }

private:
  bool useMeta = false;
};

struct ConvertBrgemmDispatchOp : public OpRewritePattern<BrgemmDispatchOp> {
  ConvertBrgemmDispatchOp(MLIRContext *context, bool useMeta,
                          PatternBenefit benefit = 1)
      : OpRewritePattern<BrgemmDispatchOp>(context, benefit), useMeta(useMeta) {
  }

  LogicalResult matchAndRewrite(BrgemmDispatchOp dispatchOp,
                                PatternRewriter &rewriter) const override {
    return buildDispatchOp<BrgemmDispatchOp>(rewriter, dispatchOp,
                                             "xsmm_brgemm_dispatch", useMeta);
  }

private:
  bool useMeta = false;
};

struct ConvertTernaryDispatchOp : public OpRewritePattern<TernaryDispatchOp> {
  ConvertTernaryDispatchOp(MLIRContext *context, bool useMeta,
                           PatternBenefit benefit = 1)
      : OpRewritePattern<TernaryDispatchOp>(context, benefit),
        useMeta(useMeta) {}

  LogicalResult matchAndRewrite(TernaryDispatchOp dispatchOp,
                                PatternRewriter &rewriter) const override {
    return buildDispatchOp<TernaryDispatchOp>(rewriter, dispatchOp,
                                              "xsmm_ternary_dispatch", useMeta);
  }

private:
  bool useMeta = false;
};

struct ConvertQuarternaryDispatchOp
    : public OpRewritePattern<QuarternaryDispatchOp> {
  ConvertQuarternaryDispatchOp(MLIRContext *context, bool useMeta,
                               PatternBenefit benefit = 1)
      : OpRewritePattern<QuarternaryDispatchOp>(context, benefit),
        useMeta(useMeta) {}

  LogicalResult matchAndRewrite(QuarternaryDispatchOp dispatchOp,
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
    dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
        loc, integer64, dispatchOp.getDataTypeAttr()));
    dispatchOperandTypes.push_back(integer64);

    BoolAttr isVNNIAttr = rewriter.getBoolAttr(dispatchOp.getIsVNNI());
    IntegerType boolType = IntegerType::get(rewriter.getContext(), 1);
    dispatchOperands.push_back(
        rewriter.create<arith::ConstantOp>(loc, boolType, isVNNIAttr));
    dispatchOperandTypes.push_back(boolType);

    ArrayRef<int64_t> integers = dispatchOp.getInputsAttr().asArrayRef();
    size_t arrayAttrSize = integers.size();
    for (size_t idx = 0; idx < arrayAttrSize; idx++) {
      IntegerAttr attr = IntegerAttr::get(rewriter.getI64Type(), integers[idx]);
      dispatchOperands.push_back(
          rewriter.create<arith::ConstantOp>(loc, integer64, attr));
      dispatchOperandTypes.push_back(integer64);
    }
    func::CallOp call =
        buildDispatchCall(loc, dispatchOperands, dispatchOperandTypes, module,
                          fnName, useMeta, rewriter);
    rewriter.replaceOp(dispatchOp, call.getResult(0));
    return success();
  }

private:
  bool useMeta = false;
};

struct ConvertBinaryDispatchOp : public OpRewritePattern<BinaryDispatchOp> {
  ConvertBinaryDispatchOp(MLIRContext *context, bool useMeta,
                          PatternBenefit benefit = 1)
      : OpRewritePattern<BinaryDispatchOp>(context, benefit), useMeta(useMeta) {
  }

  LogicalResult matchAndRewrite(BinaryDispatchOp dispatchOp,
                                PatternRewriter &rewriter) const override {
    return buildDispatchOp<BinaryDispatchOp>(rewriter, dispatchOp,
                                             "xsmm_binary_dispatch", useMeta);
  }

private:
  bool useMeta = false;
};

struct ConvertUnaryDispatchOp : public OpRewritePattern<UnaryDispatchOp> {
  ConvertUnaryDispatchOp(MLIRContext *context, bool useMeta,
                         PatternBenefit benefit = 1)
      : OpRewritePattern<UnaryDispatchOp>(context, benefit), useMeta(useMeta) {}

  LogicalResult matchAndRewrite(UnaryDispatchOp dispatchOp,
                                PatternRewriter &rewriter) const override {
    return buildDispatchOp<UnaryDispatchOp>(rewriter, dispatchOp,
                                            "xsmm_unary_dispatch", useMeta);
    return success();
  }

private:
  bool useMeta = false;
};

struct ConvertXsmmToFunc : public ConvertXsmmToFuncBase<ConvertXsmmToFunc> {
  ConvertXsmmToFunc() = default;
  ConvertXsmmToFunc(bool useExtractMetaData) {
    this->useExtractMetaData = useExtractMetaData;
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    tpp::populateXsmmToFuncPatterns(patterns, useExtractMetaData);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // namespace

void mlir::tpp::populateXsmmToFuncPatterns(RewritePatternSet &patterns,
                                           bool useExtractMetaData) {
  patterns
      .add<ConvertQuarternaryXsmmOp, ConvertTernaryXsmmOp, ConvertBinaryXsmmOp,
           ConvertUnaryXsmmOp, ConvertGemmXsmmOp, ConvertBrgemmXsmmOp>(
          patterns.getContext(), useExtractMetaData);
  patterns.add<ConvertQuarternaryDispatchOp, ConvertTernaryDispatchOp,
               ConvertBinaryDispatchOp, ConvertUnaryDispatchOp,
               ConvertGemmDispatchOp, ConvertBrgemmDispatchOp>(
      patterns.getContext(), useExtractMetaData);
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::tpp::createConvertXsmmToFuncPass() {
  return std::make_unique<ConvertXsmmToFunc>();
}
