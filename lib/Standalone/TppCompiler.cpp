//===- TppCompilerPipeline.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/TppPasses.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MathToLibm/MathToLibm.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "Standalone/TppPasses.h.inc"

namespace {

struct TppCompilerPipeline
    : public TppCompilerPipelineBase<TppCompilerPipeline> {
  void runOnOperation() override;
};

void TppCompilerPipeline::runOnOperation() {
  OpPassManager pm("builtin.module");
  pm.addNestedPass<func::FuncOp>(createMapLinalgToTppPass());
  if (enablePreconditions)
    pm.addNestedPass<func::FuncOp>(createTppEnforcePreconditions());
  pm.addPass(func::createFuncBufferizePass());
  pm.addNestedPass<func::FuncOp>(createLinalgBufferizePass());
  pm.addPass(arith::createConstantBufferizePass());
  pm.addNestedPass<func::FuncOp>(createTensorBufferizePass());
  if (enablePreconditions)
    pm.addNestedPass<func::FuncOp>(
        createConvertLinalgToTppPass(/*enabledPreconditions*/ true));
  else
    pm.addNestedPass<func::FuncOp>(createConvertLinalgToTppPass());
  pm.addNestedPass<func::FuncOp>(
      mlir::bufferization::createFinalizingBufferizePass());

  pm.addNestedPass<func::FuncOp>(createTppToLoopsPass());
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  pm.addNestedPass<func::FuncOp>(createConvertVectorToSCFPass());
  pm.addNestedPass<func::FuncOp>(createConvertSCFToCFPass());
  pm.addPass(createConvertVectorToLLVMPass());
  pm.addPass(createMemRefToLLVMPass());
  pm.addNestedPass<func::FuncOp>(createConvertMathToLLVMPass());
  pm.addPass(createConvertMathToLibmPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(createReconcileUnrealizedCastsPass());

  if (failed(runPipeline(pm, getOperation())))
    signalPassFailure();
  return;
}

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::tpp::createTppCompilerPipeline() {
  return std::make_unique<TppCompilerPipeline>();
}
