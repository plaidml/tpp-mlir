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
#include "TPP/Dialect/Transform/LinalgXTransformOps.h"
#include "TPP/Dialect/XeGPU/IR/XeGPUOps.h"
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

llvm::cl::list<int64_t> wmmaTileSizes(
    "wmma-tile-sizes", llvm::cl::desc("GPU WMMA tile sizes MxNxK"),
    llvm::cl::list_init<int64_t>(SmallVector<int64_t>{16, 16, 16}),
    llvm::cl::CommaSeparated);

llvm::cl::list<int64_t>
    gpuBlockTile("gpu-block-tile", llvm::cl::desc("GPU block tile size"),
                 llvm::cl::list_init<int64_t>(SmallVector<int64_t>{128, 128}),
                 llvm::cl::CommaSeparated);

llvm::cl::list<int64_t>
    gpuThreadTile("gpu-thread-tile", llvm::cl::desc("GPU thread tile size"),
                  llvm::cl::list_init<int64_t>(SmallVector<int64_t>{32, 32}),
                  llvm::cl::CommaSeparated);

llvm::cl::opt<int64_t> kTile("k-tile", llvm::cl::desc("GEMM K dim tiling size"),
                             llvm::cl::init(32));

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_GPUPIPELINE
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

enum class GpuType {
  Cuda,
  Vulkan,
  Intel,
};

GpuType parseGpuOption(StringRef gpuStr) {
  auto type = llvm::StringSwitch<std::optional<GpuType>>(gpuStr)
                  .CaseLower("cuda", GpuType::Cuda)
                  .CaseLower("vulkan", GpuType::Vulkan)
                  .CaseLower("intel", GpuType::Intel)
                  .Default(std::nullopt);
  assert(type && "Unsupported GPU backend");

  return *type;
}

struct GpuOptions {
  std::string triple;
  std::string chip;
  std::string features;
};

GpuOptions getGpuOptions(GpuType gpuType) {
  GpuOptions options;

  switch (gpuType) {
  case GpuType::Cuda: {
    options.triple = "nvptx64-nvidia-cuda";
    options.chip = "sm_70";
    options.features = "+ptx60";
    break;
  }
  case GpuType::Vulkan:
  case GpuType::Intel: {
    // No options needed at the moment.
    break;
  }
  }

  return options;
}

// GPU pipeline - map and lower operations to enable execution on a GPU.
struct GpuPipeline : public tpp::impl::GpuPipelineBase<GpuPipeline>,
                     UtilityPassBase<ModuleOp> {
  using GpuPipelineBase::GpuPipelineBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<gpu::GPUDialect>();
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<NVVM::NVVMDialect>();
    registry.insert<nvgpu::NVGPUDialect>();
    registry.insert<bufferization::BufferizationDialect>();
    registry.insert<spirv::SPIRVDialect>();
    registry.insert<imex::xegpu::XeGPUDialect>();
    linalgx::registerTransformDialectExtension(registry);
    check::registerBufferizableOpInterfaceExternalModels(registry);
    perf::registerBufferizableOpInterfaceExternalModels(registry);

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
    GpuOptions gpuOptions = getGpuOptions(gpuType);

    // Tile to split the kernel into threads and blocks.
    // Use default tiling to handle both packed and unpacked ops.
    pm.addPass(createCleanup());
    if (gpuType == GpuType::Intel) {
      // First split computation into grid with blocks of specified size.
      TileConsumerAndFuseProducersOptions blockTileOptions;
      blockTileOptions.tileSizes = gpuBlockTile;
      blockTileOptions.minTileFactor = 1;
      pm.addPass(createTileConsumerAndFuseProducers(blockTileOptions));

      // Then try to further split computation into subtiles.
      // This allows to split larger computations across multiple
      // threads/workitems. For smaller workloads, it provides another
      // chance for outlining.
      TileConsumerAndFuseProducersOptions threadTileOptions;
      threadTileOptions.tileSizes = gpuThreadTile;
      threadTileOptions.minTileFactor = 1;
      pm.addPass(createTileConsumerAndFuseProducers(threadTileOptions));
    } else {
      TileConsumerAndFuseProducersOptions tilingOptions;
      tilingOptions.minTileFactor = 1;
      pm.addPass(createTileConsumerAndFuseProducers(tilingOptions));
    }
    pm.addPass(createCleanup());

    // Preprocess and bufferize as further conversion requires memref
    // abstraction.
    pm.addPass(createLowerPacksAndUnPacks());
    bool dealloc = gpuType != GpuType::Cuda;
    pm.addPass(createBufferize(BufferizeOptions{dealloc}));
    pm.addPass(createConvertForAllToParallelOp());
    pm.addNestedPass<func::FuncOp>(createCleanup());

    // Convert to generic GPU ops.
    pm.addPass(createGpuConversion(GpuConversionOptions{
        gpuWmma, wmmaTileSizes, gpuType == GpuType::Intel, kTile}));

    // Lower GPU ops to the chosen GPU backend.
    switch (gpuType) {
    case GpuType::Cuda: {
      // Perform explicit GPU data transfers only for CUDA as the unified
      // memory is not currently used here.
      // Vulkan runner assumes usage of GPU unified memory.
      pm.addNestedPass<func::FuncOp>(createGpuDataTransfer());
      pm.addPass(createGpuToCuda(GpuToCudaOptions{
          gpuOptions.triple, gpuOptions.chip, gpuOptions.features}));
      break;
    }
    case GpuType::Vulkan: {
      pm.addPass(createGpuToVulkan());
      break;
    }
    case GpuType::Intel:
      pm.addPass(createXegpuFoldMemRef());
      break;
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
