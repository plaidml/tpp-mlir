//===- ConvertVectorToXsmm.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Conversion/ConvertVectorToXsmm/ConvertVectorToXsmm.h"
#include "TPP/Dialect/Xsmm/XsmmUtils.h"
#include "TPP/Transforms/Transforms.h"
#include "TPP/Transforms/Utils/TransformUtils.h"
#include "TPP/Transforms/Utils/VNNIUtils.h"
#include "TPP/Transforms/Utils/ValueUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/Transform/PDLExtension/PDLExtensionOps.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
using namespace mlir;
using namespace mlir::vector;
using namespace mlir::linalg;
using namespace mlir::func;

#include "ConvertVectorToXsmmPDLLPatterns.h.inc"

#define DEBUG_TYPE "convert-vector-to-xsmm"

FailureOr<xsmm::BrgemmInfo>
computeBrgemmInfo(PatternRewriter &rewriter, Operation *contractOp,
                  Operation *input0, Operation *input1, Operation *input2) {
  SmallVector<Value> inputs;
  LLVM_DEBUG(llvm::dbgs() << "computebrgemminfo\n");

  inputs.push_back(input0->getResult(0));
  inputs.push_back(input1->getResult(0));
  inputs.push_back(input2->getResult(0));

  SmallVector<Value> outputs;
  outputs.push_back(nullptr);
  auto failedOrbrgemmInfo = mlir::xsmm::utils::isMappableToBrgemm(
      rewriter, dyn_cast<mlir::vector::ContractionOp>(contractOp), inputs,
      outputs,
      dyn_cast<mlir::vector::ContractionOp>(contractOp).getIndexingMapsArray());
  if (failed(failedOrbrgemmInfo))
    return failure();
  xsmm::BrgemmInfo brgemmInfo = *failedOrbrgemmInfo;
  return brgemmInfo;
}

static std::pair<Operation *, Operation *>
buildBrgemm(PatternRewriter &rewriter, Operation *contractOp, Value input0,
            Value input1, Value input2, xsmm::BrgemmInfo brgemmInfo,
            SmallVector<Attribute> flags) {
  SmallVector<Value> inputs;
  inputs.push_back(input0);
  inputs.push_back(input1);
  inputs.push_back(input2);
  auto m = brgemmInfo.m;
  auto n = brgemmInfo.n;
  auto k = brgemmInfo.k;
  auto batch = brgemmInfo.batch;
  int64_t lda = brgemmInfo.lda;
  int64_t ldb = brgemmInfo.ldb;
  int64_t ldc = brgemmInfo.ldc;
  int64_t strideA = brgemmInfo.strideA;
  int64_t strideB = brgemmInfo.strideB;
  auto loc = contractOp->getLoc();
  auto functionOp = contractOp->getParentOfType<func::FuncOp>();
  auto dtype =
      xsmm::utils::getDataType(rewriter, contractOp->getOperand(0).getType());
  IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
  SmallVector<Value, 10> dispatchOperands;
  SmallVector<Type, 10> dispatchOperandTypes;
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(&*functionOp.getBody().op_begin());
  // Dispatch the data type.
  dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
      loc, integer64, cast<TypedAttr>(dtype)));
  dispatchOperandTypes.push_back(integer64);

  ArrayAttr brgemmFlags = rewriter.getArrayAttr(flags);
  SmallVector<Value, 10> invokeOperands;
  std::string dispatchName = "xsmm_gemm_dispatch";
  std::string invokeName = "xsmm_gemm_invoke";

  if (batch != 0) {
    dispatchName = "xsmm_brgemm_dispatch";
    invokeName = "xsmm_brgemm_invoke";
  }

  auto dims = SmallVector<int64_t>{m, n, k, lda, ldb, ldc};
  for (size_t idx = 0; idx < dims.size(); idx++) {
    dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
        loc, integer64, rewriter.getIntegerAttr(integer64, dims[idx])));
    dispatchOperandTypes.push_back(integer64);
  }
  // Dispatch the flags. Pass to the library the already ored-flag to
  // avoid changing the interface every time we add a new flag. Flags
  // are assumed to be verified before (i.e., op verifier).
  if (batch != 0) {
    dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
        loc, integer64, rewriter.getIntegerAttr(integer64, strideA)));
    dispatchOperandTypes.push_back(integer64);

    dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
        loc, integer64, rewriter.getIntegerAttr(integer64, strideB)));
    dispatchOperandTypes.push_back(integer64);
  }
  int64_t oredFlag = xsmm::utils::getOredFlags(brgemmFlags);
  dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
      loc, integer64, IntegerAttr::get(rewriter.getI64Type(), oredFlag)));
  dispatchOperandTypes.push_back(integer64);

  ModuleOp module = contractOp->getParentOfType<ModuleOp>();
  auto dispatched = xsmm::utils::buildDispatchCall(
      rewriter, loc, dispatchOperands, dispatchOperandTypes, module,
      SymbolRefAttr::get(contractOp->getContext(), dispatchName));
  SmallVector<Value, 6> operandRange;
  operandRange.push_back(rewriter.create<arith::ConstantOp>(
      loc, integer64, cast<TypedAttr>(dtype)));
  operandRange.push_back(dispatched.getResult(0));

  SmallVector<Value> results;
  results.push_back(input0.getDefiningOp()->getOperand(0));
  results.push_back(input1.getDefiningOp()->getOperand(0));
  results.push_back(input2.getDefiningOp()->getOperand(0));

  if (batch != 0) {
    Value batchDim = rewriter.create<arith::ConstantOp>(
        loc, integer64, rewriter.getIntegerAttr(integer64, batch));
    results.push_back(batchDim);
  }
  SmallVector<Value> preceedingOperands;
  auto invokeCall = xsmm::utils::buildInvokeCall(
      rewriter, contractOp, module, results, preceedingOperands, -1,
      operandRange, invokeName, dtype);
  return std::make_pair(&*dispatched, &*invokeCall);
}

static std::pair<Operation *, Operation *>
buildOpWithBetaZeroImpl(PatternRewriter &rewriter, Operation *contractOp,
                        Operation *input0, Operation *input1, Operation *input2,
                        Operation *betaZero) {
  LLVM_DEBUG(llvm::dbgs() << "buildOpWithBetaZeroImpl\n");
  auto brgemmInfo =
      computeBrgemmInfo(rewriter, contractOp, input0, input1, input2);
  SmallVector<Attribute> flags;
  if (brgemmInfo->isVnni) {
    flags.push_back(xsmm::GemmFlagsAttr::get(rewriter.getContext(),
                                             xsmm::GemmFlags::VNNI_B));
  }
  flags.push_back(
      xsmm::GemmFlagsAttr::get(rewriter.getContext(), xsmm::GemmFlags::BETA_0));

  return buildBrgemm(rewriter, contractOp, input0->getResult(0),
                     input1->getResult(0), input2->getResult(0), *brgemmInfo,
                     flags);
}

static std::pair<Operation *, Operation *>
buildOpImpl(PatternRewriter &rewriter, Operation *contractOp, Operation *input0,
            Operation *input1, Operation *input2) {
  LLVM_DEBUG(llvm::dbgs() << "buildOpImpl\n");
  FailureOr<mlir::xsmm::BrgemmInfo> brgemmInfo;
  brgemmInfo = computeBrgemmInfo(rewriter, contractOp, input0, input1, input2);
  SmallVector<Attribute> flags;
  if (brgemmInfo->isVnni) {
    flags.push_back(xsmm::GemmFlagsAttr::get(rewriter.getContext(),
                                             xsmm::GemmFlags::VNNI_B));
  }

  return buildBrgemm(rewriter, contractOp, input0->getResult(0),
                     input1->getResult(0), input2->getResult(0), *brgemmInfo,
                     flags);
}

static LogicalResult validateOpImpl(PatternRewriter &rewriter,
                                    Operation *contractOp, Operation *input0,
                                    Operation *input1, Operation *input2,
                                    Operation *result) {
  LLVM_DEBUG(llvm::dbgs() << "validateOpImpl\n");
  FailureOr<xsmm::BrgemmInfo> brgemmInfo =
      computeBrgemmInfo(rewriter, contractOp, input0, input1, input2);

  if (failed(brgemmInfo)) {
    return failure(contractOp);
  }

  return success(contractOp);
}

void registerNativeRewrite(RewritePatternSet &patterns) {
  patterns.getPDLPatterns().registerRewriteFunction("BuildOpWithBetaZero",
                                                    buildOpWithBetaZeroImpl);
  patterns.getPDLPatterns().registerRewriteFunction("BuildOp", buildOpImpl);
  patterns.getPDLPatterns().registerConstraintFunction("ValidateOp",
                                                       validateOpImpl);
}

namespace mlir {
namespace tpp {

struct ConvertVectorToXsmm
    : public PassWrapper<ConvertVectorToXsmm, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertVectorToXsmm)

  StringRef getArgument() const final { return "convert-vector-to-xsmm-pass"; }

  StringRef getDescription() const final {
    return "Convert vector to XSMM functionality";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<pdl::PDLDialect, pdl_interp::PDLInterpDialect,
                    mlir::vector::VectorDialect, func::FuncDialect,
                    memref::MemRefDialect, LLVM::LLVMDialect, BuiltinDialect>();
  }

  LogicalResult initialize(MLIRContext *ctx) override {
    // Build the pattern set within the `initialize` to avoid recompiling
    // PDL patterns during each `runOnOperation` invocation.
    RewritePatternSet patternList(ctx);
    registerNativeRewrite(patternList);
    populateGeneratedPDLLPatterns(patternList);
    patterns = std::move(patternList);
    return success();
  }

  void runOnOperation() final {
    PatternRewriter rewriter(&getContext());
    // Enable conversion for linalg.generic to XSMM Brgemm if possible.
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patterns))) {
      signalPassFailure();
    }
  }

  FrozenRewritePatternSet patterns;
};

std::unique_ptr<mlir::Pass> createConvertVectorToXsmm() {
  return std::make_unique<ConvertVectorToXsmm>();
}

} // namespace tpp
} // namespace mlir
