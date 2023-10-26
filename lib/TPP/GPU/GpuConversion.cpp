//===- GpuConversion.cpp -----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "TPP/Dialect/Tpp/TppDialect.h"
#include "TPP/Dialect/Transform/LinalgXTransformOps.h"
#include "TPP/PassUtils.h"

#include <optional>

using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

// Map and lower operations to generic GPU ops.
struct GpuConversion : public GpuConversionBase<GpuConversion>,
                       UtilityPassBase<ModuleOp> {
  GpuConversion() = default;
  GpuConversion(bool useWmma) { this->useWmma = useWmma; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tpp::TppDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<gpu::GPUDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();

    // Initialize the pipeline if needed.
    // Otherwise, just run the cached one.
    if (pm.empty())
      constructPipeline();

    if (failed(runPipeline(pm, module)))
      return signalPassFailure();
  }

private:
  void constructPipeline() override {
    pm.clear();

    // Lower TPP ops to GPU-compatible format.
    ConvertTppToLoopsOptions tppToLoopOptions;
    tppToLoopOptions.parallel = true;
    pm.addNestedPass<func::FuncOp>(createConvertTppToLoops(tppToLoopOptions));

    // First lower linalg using custom patterns then fall back to
    // the default lowering for any remaining ops.
    pm.addNestedPass<func::FuncOp>(createLinalgDeGeneralize());
    pm.addNestedPass<func::FuncOp>(createLinalgToGpuPass(useWmma));
    pm.addNestedPass<func::FuncOp>(createConvertLinalgToParallelLoopsPass());

    // Map loops into GPU kernels.
    pm.addNestedPass<func::FuncOp>(createGpuMapParallelLoopsPass());
    pm.addNestedPass<func::FuncOp>(createParallelLoopToGpuPass());

    pm.addNestedPass<func::FuncOp>(createCleanup());

    // Create GPU kernels.
    pm.addPass(createGpuKernelOutliningPass());

    // Generic cleanup.
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::tpp::createGpuConversionPass(bool useWmma) {
  return std::make_unique<GpuConversion>(useWmma);
}
