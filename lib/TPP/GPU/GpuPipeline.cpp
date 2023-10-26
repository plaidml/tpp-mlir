//===- GpuPipeline.cpp -------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
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
#include "TPP/PassUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#include <optional>

using namespace mlir;
using namespace mlir::tpp;

llvm::cl::opt<bool> gpuWmma("gpu-wmma",
                            llvm::cl::desc("Enable GPU WMMA support"),
                            llvm::cl::init(false));

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

enum class GpuType {
  Cuda,
  Vulkan,
};

GpuType parseGpuOption(StringRef gpuStr) {
  auto type = llvm::StringSwitch<std::optional<GpuType>>(gpuStr)
                  .CaseLower("cuda", GpuType::Cuda)
                  .CaseLower("vulkan", GpuType::Vulkan)
                  .Default(std::nullopt);
  assert(type && "Unsupported GPU backend");

  return *type;
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
    registry.insert<bufferization::BufferizationDialect>();
    registry.insert<spirv::SPIRVDialect>();
    linalgx::registerTransformDialectExtension(registry);
    check::registerBufferizableOpInterfaceExternalModels(registry);
    perf::registerBufferizableOpInterfaceExternalModels(registry);
    tpp::registerBufferizableOpInterfaceExternalModels(registry);

    // Add all core MLIR dialects to make the pipeline more robust with respect
    // to accepted input IR by preventing cryptic runtime crashes due to missing
    // dialect registrations.
    registerAllDialects(registry);
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

    GpuType gpuType = parseGpuOption(this->gpuBackend);

    // Tile to split the kernel into threads and blocks.
    // Use default tiling to handle both packed and unpacked ops.
    pm.addPass(createCleanup());
    pm.addPass(createTileConsumerAndFuseProducers());
    pm.addPass(createCleanup());

    // Preprocess and bufferize as further conversion requires memref
    // abstraction.
    pm.addPass(createLowerPacksAndUnPacks());
    bool dealloc = gpuType != GpuType::Cuda;
    pm.addPass(createBufferize(BufferizeOptions{dealloc}));
    pm.addPass(createConvertForAllToParallelOp());
    pm.addNestedPass<func::FuncOp>(createCleanup());

    // Convert to generic GPU ops.
    pm.addPass(createGpuConversionPass(gpuWmma));

    // Lower GPU ops to the chosen GPU backend.
    switch (gpuType) {
    case GpuType::Cuda: {
      std::string gpuTriple = "nvptx64-nvidia-cuda";
      std::string gpuChip = "sm_70";
      std::string gpuFeatures = "+ptx60";

      // Perform explicit GPU data transfers only for CUDA as the unified
      // memory is not currently used here.
      // Vulkan runner assumes usage of GPU unified memory.
      pm.addNestedPass<func::FuncOp>(createGpuDataTransfer());
      pm.addPass(createGpuToCudaPass(gpuTriple, gpuChip, gpuFeatures));
      break;
    }
    case GpuType::Vulkan: {
      pm.addPass(createGpuToVulkanPass());
      break;
    }
    }

    // Clean up after the GPU pipeline.
    // Use upstream passes directly instead of the cleanup pass as the GPU
    // kernel is at the LLVM dialect level which is not compatible with the
    // custom TPP passes.
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::tpp::createGpuPipelinePass(StringRef gpuBackend) {
  return std::make_unique<GpuPipeline>(gpuBackend);
}
