//===- TppCompilerPipeline.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MathToLibm/MathToLibm.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
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
#include "TPP/Passes.h.inc"

namespace {

struct TppCompilerPipeline
    : public TppCompilerPipelineBase<TppCompilerPipeline> {
  void runOnOperation() override;
};

void TppCompilerPipeline::runOnOperation() {
  OpPassManager pm("builtin.module");
  // map-linalg-to-tpp
  pm.addNestedPass<func::FuncOp>(createMapLinalgToTppPass());
  // enforce-tpp-preconditions
  if (enablePreconditions)
    pm.addNestedPass<func::FuncOp>(createPasSIMDDimensionPass());
  // func-bufferize
  pm.addPass(func::createFuncBufferizePass());
  // linalg-bufferize
  pm.addNestedPass<func::FuncOp>(createLinalgBufferizePass());
  // arith-bufferize
  pm.addPass(arith::createConstantBufferizePass());
  // tensor-bufferize
  pm.addNestedPass<func::FuncOp>(createTensorBufferizePass());
  // convert-linalg-to-tpp
  if (enablePreconditions)
    pm.addNestedPass<func::FuncOp>(createConvertLinalgToTppPass(
        /*enabledPreconditions*/ true, /*useParallelLoops*/ true));
  else
    pm.addNestedPass<func::FuncOp>(createConvertLinalgToTppPass());
  // finalizing-bufferize
  pm.addNestedPass<func::FuncOp>(
      mlir::bufferization::createFinalizingBufferizePass());

  // remove-extra-copies
  // pm.addNestedPass<func::FuncOp>(createCopyRemovalPass());

  // ----

  // Another round of detection to increase mapping to tpp.
  // (e.g., linalg.fill -> tpp.identity)
  pm.addNestedPass<func::FuncOp>(createLinalgGeneralizationPass());
  pm.addNestedPass<func::FuncOp>(createMapLinalgToTppPass());
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToTppPass());
  pm.addNestedPass<func::FuncOp>(createVectorizeCopyPass());

  // -----

  if (enableXsmmConversion) // convert-tpp-to-xsmm
    pm.addNestedPass<func::FuncOp>(createConvertTppToXsmmPass());
  else // convert-tpp-to-loops
    pm.addNestedPass<func::FuncOp>(createConvertTppToLoopsPass());

  pm.addPass(createConvertXsmmToFuncPass());
  pm.addPass(createConvertCheckToFuncPass());
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  pm.addNestedPass<func::FuncOp>(arith::createArithExpandOpsPass());
  pm.addNestedPass<func::FuncOp>(createConvertVectorToSCFPass());
  pm.addNestedPass<func::FuncOp>(createConvertSCFToCFPass());
  pm.addPass(createConvertVectorToLLVMPass());
  pm.addNestedPass<func::FuncOp>(createConvertMathToLLVMPass());
  pm.addPass(createConvertMathToLibmPass());
  pm.addPass(createConvertFuncToLLVMPass());
  // pm.addPass(createMemRefToLLVMPass());
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
