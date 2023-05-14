//===- Bufferize.cpp ---------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"
#include "TPP/Transforms.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "TPP/Dialect/Check/BufferizableOpInterfaceImpl.h"
#include "TPP/Dialect/Check/CheckDialect.h"
#include "TPP/Dialect/Perf/BufferizableOpInterfaceImpl.h"
#include "TPP/Dialect/Perf/PerfDialect.h"
#include "TPP/Dialect/Tpp/BufferizableOpInterfaceImpl.h"
#include "TPP/Dialect/Tpp/TppDialect.h"
#include "TPP/Dialect/Transform/LinalgXTransformOps.h"
#include "TPP/Dialect/Xsmm/XsmmDialect.h"

using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

struct Bufferize : public BufferizeBase<Bufferize> {
  Bufferize() = default;
  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry
        .insert<affine::AffineDialect,
                arith::ArithDialect,
                bufferization::BufferizationDialect,
                func::FuncDialect,
                linalg::LinalgDialect,
                memref::MemRefDialect,
                check::CheckDialect,
                perf::PerfDialect,
                scf::SCFDialect,
                tpp::TppDialect,
                tensor::TensorDialect>();
    // clang-format on
    check::registerBufferizableOpInterfaceExternalModels(registry);
    perf::registerBufferizableOpInterfaceExternalModels(registry);
    tpp::registerBufferizableOpInterfaceExternalModels(registry);
  }
  void runOnOperation() override;
};

void Bufferize::runOnOperation() {
  ModuleOp moduleOp = getOperation();

  OpPassManager passManager;

  // Pre-processing.
  passManager.addPass(bufferization::createEmptyTensorToAllocTensorPass());

  // One-shot.
  bufferization::OneShotBufferizationOptions buffOpts;
  buffOpts.allowReturnAllocs = true;
  buffOpts.bufferizeFunctionBoundaries = true;
  buffOpts.setFunctionBoundaryTypeConversion(
      bufferization::LayoutMapOption::IdentityLayoutMap);
  bool runOnlyAnalysis = this->testAnalysisOnly || this->printConflicts;
  if (runOnlyAnalysis) {
    buffOpts.printConflicts = this->printConflicts;
    buffOpts.testAnalysisOnly = this->testAnalysisOnly;
  }
  passManager.addPass(bufferization::createOneShotBufferizePass(buffOpts));

  if (!runOnlyAnalysis) {
    passManager.addPass(bufferization::createDropEquivalentBufferResultsPass());
    passManager.addNestedPass<func::FuncOp>(
        bufferization::createFinalizingBufferizePass());

    // Post-processing.
    passManager.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    passManager.addNestedPass<func::FuncOp>(createCSEPass());
    // There are redundant memcpy (with linalg.generic form) ops created, which
    // can be deleted by canonicalizer. We have to run it again because the
    // memrefs are unified in CSE pass, so we can truly remove redundant memcpy.
    passManager.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  }

  if (failed(runPipeline(passManager, moduleOp)))
    return signalPassFailure();
}

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::tpp::createBufferizePass() {
  return std::make_unique<Bufferize>();
}
