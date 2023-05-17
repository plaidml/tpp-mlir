//===- GpuPipeline.cpp -------------------------------------------*- C++-*-===//
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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "TPP/Dialect/Tpp/TppDialect.h"
#include "TPP/Dialect/Transform/LinalgXTransformOps.h"
#include "TPP/PassUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

enum class GpuType {
  None, // no target GPU
  Cuda,
};

GpuType parseGpuOption(StringRef gpuStr) {
  return llvm::StringSwitch<GpuType>(gpuStr)
      .CaseLower("none", GpuType::None)
      .CaseLower("cuda", GpuType::Cuda)
      .Default(GpuType::None);
}

// GPU pipeline - map and lower operations to enable execution on a GPU.
struct GpuPipeline : public GpuPipelineBase<GpuPipeline>,
                     UtilityPassBase<ModuleOp> {
  GpuPipeline() = default;
  GpuPipeline(StringRef gpuBackend) { this->gpuBackend = gpuBackend.str(); }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tpp::TppDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<gpu::GPUDialect>();
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<NVVM::NVVMDialect>();
    registry.insert<nvgpu::NVGPUDialect>();
  }

  void runOnOperation() override {
    // Do nothing when no GPU is selected.
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

    GpuType gpuType = parseGpuOption(this->gpuBackend);

    // Add no passes when no GPU is selected.
    if (gpuType == GpuType::None)
      return;

    // Map and lower ops to GPU-compatible format.
    pm.addNestedPass<func::FuncOp>(createConvertLinalgToParallelLoopsPass());
    pm.addNestedPass<func::FuncOp>(createGpuMapParallelLoopsPass());
    pm.addNestedPass<func::FuncOp>(createParallelLoopToGpuPass());

    pm.addNestedPass<func::FuncOp>(createCleanupPass());

    // Create GPU kernels.
    pm.addPass(createGpuKernelOutliningPass());

    // Lower GPU ops to the chosen GPU backend.
    if (gpuType == GpuType::Cuda)
      pm.addNestedPass<gpu::GPUModuleOp>(createGpuToCudaPass());

    // Clean up after the GPU pipeline.
    // Use upstream passes directly instead of the cleanup pass as the GPU
    // kernel is at the LLVM dialect level which is not compatible with the
    // custom TPP passes.
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::tpp::createGpuPipelinePass() {
  return std::make_unique<GpuPipeline>();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::tpp::createGpuPipelinePass(StringRef gpuBackend) {
  return std::make_unique<GpuPipeline>(gpuBackend);
}
