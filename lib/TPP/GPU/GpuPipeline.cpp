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
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "TPP/Dialect/Tpp/BufferizableOpInterfaceImpl.h"
#include "TPP/Dialect/Tpp/TppDialect.h"
#include "TPP/Dialect/Transform/LinalgXTransformOps.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::tpp;

// Select target GPU for the pipeline.
llvm::cl::opt<std::string>
    defGpuType("gpu", llvm::cl::desc("Target GPU for the lowering"),
               llvm::cl::value_desc("none,nvidia"), llvm::cl::init("none"));

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

enum class GpuType {
  NONE,   // no target GPU
  NVIDIA, // CUDA backend
};

GpuType parseGpuOption(StringRef gpuStr) {
  return llvm::StringSwitch<GpuType>(gpuStr)
      .CaseLower("none", GpuType::NONE)
      .CaseLower("nvidia", GpuType::NVIDIA)
      .Default(GpuType::NONE);
}

template <typename OpT> class UtilityPassBase {
public:
  UtilityPassBase()
      : pm(OpT::getOperationName(), mlir::OpPassManager::Nesting::Implicit){};
  virtual ~UtilityPassBase() = default;

protected:
  OpPassManager pm;

  // Create the pass processing pipeline.
  virtual void constructPipeline() = 0;
};

struct GpuToCuda
    : public PassWrapper<GpuToCuda, OperationPass<gpu::GPUModuleOp>>,
      UtilityPassBase<gpu::GPUModuleOp> {
  GpuToCuda() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect>();
    registry.insert<NVVM::NVVMDialect>();
    registry.insert<nvgpu::NVGPUDialect>();
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

    // Create GPU kernels.
    pm.addPass(createStripDebugInfoPass());
    pm.addPass(createLowerGpuOpsToNVVMOpsPass());
    pm.addPass(createReconcileUnrealizedCastsPass());
    pm.addPass(createGpuSerializeToCubinPass());
  }
};

std::unique_ptr<OperationPass<gpu::GPUModuleOp>> createGpuToCuda() {
  return std::make_unique<GpuToCuda>();
}

// GPU pipeline - map and lower operations to enable execution on a GPU.
struct GpuPipeline : public GpuPipelineBase<GpuPipeline>,
                     UtilityPassBase<ModuleOp> {
  GpuPipeline() { this->gpuType = parseGpuOption(defGpuType); }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tpp::TppDialect>();
    registry.insert<gpu::GPUDialect>();
    bufferization::registerAllocationOpInterfaceExternalModels(registry);
    linalgx::registerTransformDialectExtension(registry);
    tpp::registerBufferizableOpInterfaceExternalModels(registry);

    // Add all core MLIR dialects as the default TPP passes may contain any
    // combination of other passes.
    registerAllDialects(registry);
  }

  void runOnOperation() override {
    // Do nothing when no GPU is selected.
    if (gpuType != GpuType::NONE) {
      auto module = getOperation();

      // Initialize the pipeline if needed.
      // Otherwise, just run the cached one.
      if (pm.empty())
        constructPipeline();

      if (failed(runPipeline(pm, module)))
        return signalPassFailure();
    }
  }

private:
  GpuType gpuType;

  void constructPipeline() override {
    pm.clear();

    // Add no passes when no GPU is selected.
    if (gpuType == GpuType::NONE)
      return;

    // Map and lower ops to GPU-compatible format.
    pm.addNestedPass<func::FuncOp>(createConvertLinalgToParallelLoopsPass());
    pm.addNestedPass<func::FuncOp>(createGpuMapParallelLoopsPass());
    pm.addNestedPass<func::FuncOp>(createParallelLoopToGpuPass());

    pm.addNestedPass<func::FuncOp>(createCleanupPass());

    // Create GPU kernels.
    pm.addPass(createGpuKernelOutliningPass());

    // Lower GPU to CUDA backend.
    pm.addNestedPass<gpu::GPUModuleOp>(createGpuToCuda());

    // Finalize GPU lowering.
    pm.addPass(createGpuToLLVMConversionPass());

    // Clean up after the GPU pipeline.
    pm.addNestedPass<func::FuncOp>(createCleanupPass());
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::tpp::createGpuPipeline() {
  return std::make_unique<GpuPipeline>();
}
