//===- ConvertPerfToFunc.cpp -------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Perf/PerfOps.h"
#include "TPP/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::perf;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

// Perf dialect type normalization helper function.
// Cast memref and tensor to their unranked versions and convert
// custom perf types into default primitive types.
// Leave all the other operands as they are.
static SmallVector<Type> extractNormalizedTypes(OpBuilder &b,
                                                ValueRange values) {
  SmallVector<Type> results;
  results.reserve(values.size());

  for (Value val : values) {
    TypeSwitch<Type>(val.getType())
        .Case<MemRefType>([&](Type t) {
          auto memrefType = cast<MemRefType>(t);
          auto unrankedMemref = UnrankedMemRefType::get(
              memrefType.getElementType(), memrefType.getMemorySpace());
          results.push_back(unrankedMemref);
        })
        .Case<TensorType>([&](Type t) {
          auto tensorType = cast<TensorType>(t);
          auto unrankedTensor =
              UnrankedTensorType::get(tensorType.getElementType());
          results.push_back(unrankedTensor);
        })
        .Case<TimerType>([&](Type t) {
          auto i64 = IntegerType::get(b.getContext(), 64);
          results.push_back(i64);
        })
        .Default([&](Type t) { results.push_back(t); });
  }

  return results;
}

// Similar to 'extractNormalizedTypes' but inserts conversions from original
// types to normalized ones when possible.
// The conversions aim to simplify runtime function signature generation by, for
// example, erasing explicit memref and tensor shapes.
static SmallVector<Value> getNormalizedOperands(OpBuilder &b, Location loc,
                                                ValueRange operands) {
  SmallVector<Value> res;
  res.reserve(operands.size());

  for (Value op : operands) {
    auto type = extractNormalizedTypes(b, op);
    TypeSwitch<Type>(op.getType())
        .Case<MemRefType>([&](Type t) {
          Value cast = b.create<memref::CastOp>(loc, type, op);
          res.push_back(cast);
        })
        .Case<TensorType>([&](Type t) {
          Value cast = b.create<tensor::CastOp>(loc, type, op);
          res.push_back(cast);
        })
        .Default([&](Type t) { res.push_back(op); });
  }

  return res;
}

// Apply custom function name mangling for various data types.
// It is assumed that all relevant perf operations accept only unranked memory
// types. This allows for simpler name mangling and leaner perf runtime.
static void applyTypeMangling(std::string &name, Type type) {
  llvm::raw_string_ostream mangledName(name);

  TypeSwitch<Type>(type)
      .Case<MemRefType>([&](Type t) {
        mangledName << "_memref"
                    << "_" << cast<MemRefType>(t).getElementType();
      })
      .Case<TensorType>([&](Type t) {
        mangledName << "_tensor"
                    << "_" << cast<TensorType>(t).getElementType();
      })
      .Default([&](Type t) { mangledName << "_" << t; });
}

// Generate function implementation for perf.mean operation.
static void buildPerfMeanFunc(Location loc, func::FuncOp func, Operation *op,
                              PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);

  // Create function body
  Block *block = func.addEntryBlock();
  rewriter.setInsertionPointToEnd(block);

  // Check assumptions on function arguments.
  auto argTypes = func.getFunctionType().getInputs();
  assert((argTypes.size() == 1) && "expect only 1 function argument");
  assert((argTypes[0].isa<UnrankedMemRefType>()) &&
         "expect unranked memref argument");

  // Cast the buffer to something directly iteratable.
  auto buff = block->getArguments()[0];
  auto memRefType = MemRefType::get(
      ShapedType::kDynamic,
      buff.getType().cast<UnrankedMemRefType>().getElementType());
  auto deltas = rewriter.create<memref::CastOp>(loc, memRefType, buff);

  // Iterate over the whole buffer and sum up all time delta values.
  // Implemented directly as scf to keep further lowering simple.
  auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  auto len = rewriter.create<memref::DimOp>(loc, deltas, zero);
  auto floatType = rewriter.getF64Type();
  auto result = rewriter.create<arith::ConstantFloatOp>(
      loc, APFloat::getZero(floatType.getFloatSemantics()), floatType);

  auto loopNest = scf::buildLoopNest(
      rewriter, loc, /*lbs=*/ValueRange{zero}, /*ubs=*/ValueRange{len},
      /*steps=*/ValueRange{one}, ValueRange{result},
      [&](OpBuilder &b, Location loc, ValueRange localIvs,
          ValueRange iterArgs) -> scf::ValueVector {
        auto delta = rewriter.create<memref::LoadOp>(loc, deltas, localIvs);
        auto sum = rewriter.create<arith::AddFOp>(loc, delta, iterArgs[0]);

        return scf::ValueVector({sum});
      });
  assert((loopNest.results.size() == 1) && "expect only 1 loop result");

  // Compute average delta value.
  auto lenInt = rewriter.create<arith::IndexCastOp>(
      loc, rewriter.getIntegerType(64), len);
  auto size = rewriter.create<arith::SIToFPOp>(loc, floatType, lenInt);
  auto mean =
      rewriter.create<arith::DivFOp>(loc, floatType, loopNest.results[0], size);

  // Return the computed mean value.
  rewriter.create<func::ReturnOp>(loc, ValueRange{mean});
}

// Creates function prototypes and insert calls to the perf runtime functions.
static LogicalResult buildPerfFuncCall(Location loc, std::string funcName,
                                       Operation *op,
                                       PatternRewriter &rewriter) {
  if (op->getNumResults() > 1)
    return op->emitError(
               "expected operation to have 0 or 1 result, but provided ")
           << op->getNumResults();

  FlatSymbolRefAttr fnName = SymbolRefAttr::get(op->getContext(), funcName);
  ModuleOp module = op->getParentOfType<ModuleOp>();

  // Create function prototype if it is not available yet.
  if (!module.lookupSymbol(fnName.getAttr())) {
    OpBuilder::InsertionGuard guard(rewriter);
    // Insert before module terminator.
    rewriter.setInsertionPoint(module.getBody(),
                               std::prev(module.getBody()->end()));

    auto libFnType = rewriter.getFunctionType(
        extractNormalizedTypes(rewriter, op->getOperands()),
        extractNormalizedTypes(rewriter, op->getResults()));
    func::FuncOp funcOp =
        rewriter.create<func::FuncOp>(loc, fnName.getValue(), libFnType);
    funcOp.setPrivate();

    TypeSwitch<Operation *>(op)
        .Case<perf::MeanOp>([&](Operation *op) {
          buildPerfMeanFunc(loc, funcOp, op, rewriter);
        })
        .Default([&](Operation *op) {
          funcOp->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                          UnitAttr::get(rewriter.getContext()));
        });
  }

  // Insert a function call to the perf runtime.
  auto funcCall = rewriter.create<func::CallOp>(
      loc, fnName.getValue(),
      extractNormalizedTypes(rewriter, op->getResults()),
      getNormalizedOperands(rewriter, loc, op->getOperands()));
  op->replaceAllUsesWith(funcCall.getResults());

  return success();
}

struct ConvertStartTimerOp : public OpRewritePattern<perf::StartTimerOp> {
  using OpRewritePattern<perf::StartTimerOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(perf::StartTimerOp startTimerOp,
                                PatternRewriter &rewriter) const override {
    auto res = buildPerfFuncCall(startTimerOp.getLoc(), "perf_start_timer",
                                 startTimerOp, rewriter);
    if (succeeded(res))
      rewriter.eraseOp(startTimerOp);
    return res;
  }
};

struct ConvertStopTimerOp : public OpRewritePattern<perf::StopTimerOp> {
  using OpRewritePattern<perf::StopTimerOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(perf::StopTimerOp stopTimerOp,
                                PatternRewriter &rewriter) const override {
    auto res = buildPerfFuncCall(stopTimerOp.getLoc(), "perf_stop_timer",
                                 stopTimerOp, rewriter);
    if (succeeded(res))
      rewriter.eraseOp(stopTimerOp);
    return res;
  }
};

struct ConvertMeanOp : public OpRewritePattern<perf::MeanOp> {
  using OpRewritePattern<perf::MeanOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(perf::MeanOp meanOp,
                                PatternRewriter &rewriter) const override {
    auto res =
        buildPerfFuncCall(meanOp.getLoc(), "perf_mean", meanOp, rewriter);
    if (succeeded(res))
      rewriter.eraseOp(meanOp);
    return res;
  }
};

struct ConvertStdevOp : public OpRewritePattern<perf::StdevOp> {
  using OpRewritePattern<perf::StdevOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(perf::StdevOp stdevOp,
                                PatternRewriter &rewriter) const override {
    auto res =
        buildPerfFuncCall(stdevOp.getLoc(), "perf_stdev", stdevOp, rewriter);
    if (succeeded(res))
      rewriter.eraseOp(stdevOp);
    return res;
  }
};

struct ConvertSinkOp : public OpRewritePattern<perf::SinkOp> {
  using OpRewritePattern<perf::SinkOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(perf::SinkOp sinkOp,
                                PatternRewriter &rewriter) const override {
    std::string funcName("perf_sink");
    // perf.sink relies on the perf runtime to prevent complier from
    // optimizing away marked data. Name mangling is required as the op
    // accepts any kind of data. For simplicity, the mangling makes some
    // assumptions on data types supported by the perf dialect.
    applyTypeMangling(funcName, sinkOp.getInput().getType());

    auto res = buildPerfFuncCall(sinkOp.getLoc(), funcName, sinkOp, rewriter);
    if (succeeded(res))
      rewriter.eraseOp(sinkOp);
    return res;
  }
};

void populatePerfToFuncPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertStartTimerOp, ConvertStopTimerOp, ConvertMeanOp,
               ConvertStdevOp, ConvertSinkOp>(patterns.getContext());
}

struct ConvertPerfToFunc : public ConvertPerfToFuncBase<ConvertPerfToFunc> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populatePerfToFuncPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::tpp::createConvertPerfToFuncPass() {
  return std::make_unique<ConvertPerfToFunc>();
}
