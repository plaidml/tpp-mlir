//===- Bufferization.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "Standalone/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::bufferization;

#define GEN_PASS_CLASSES
#include "Standalone/Passes.h.inc"

namespace {

struct Bufferize : public BufferizationBase<Bufferize> {
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<arith::ArithDialect,
                    bufferization::BufferizationDialect,
                    linalg::LinalgDialect,
                    memref::MemRefDialect,
                    scf::SCFDialect,
                    tensor::TensorDialect,
                    vector::VectorDialect>();
    // clang-format on
  }
};

static LogicalResult runOneShot(ModuleOp module,
                                const OneShotBufferizationOptions &options) {
  OneShotAnalysisState state(module, options);
  if (failed(analyzeOp(module, state)))
    return failure();
  if (options.testAnalysisOnly)
    return success();
  return bufferization::runOneShotBufferize(module, options);
}

void Bufferize::runOnOperation() {
  ModuleOp module = getOperation();
  OneShotBufferizationOptions options;
  options.allowReturnAllocs = true;
  options.bufferizeFunctionBoundaries = true;
  options.functionBoundaryTypeConversion =
      BufferizationOptions::LayoutMapOption::IdentityLayoutMap;
  if (failed(runOneShot(module, options)))
    signalPassFailure();
  mlir::PassManager pm(module.getContext());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addPass(createDropEquivalentBufferResultsPass());
  pm.addPass(createBufferResultsToOutParamsPass());
  pm.addNestedPass<func::FuncOp>(createBufferDeallocationPass());
  pm.addNestedPass<func::FuncOp>(createFinalizingBufferizePass());
  if (failed(pm.run(module)))
    signalPassFailure();
}

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::tpp::createBufferizationPass() {
  return std::make_unique<Bufferize>();
}
