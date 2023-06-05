//===- GpuToVulkan.cpp -------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/RequestCWrappers.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

#include "TPP/Dialect/Tpp/TppDialect.h"
#include "TPP/PassUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

// Lower generic GPU ops to Vulkan backend.
struct GpuToVulkan : public GpuToVulkanBase<GpuToVulkan>,
                     UtilityPassBase<ModuleOp> {
  GpuToVulkan() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect>();
    registry.insert<spirv::SPIRVDialect>();
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

#ifdef TPP_GPU_ENABLE
    // Create SPIRV kernel.
    pm.addPass(createConvertGPUToSPIRVPass());
    pm.addNestedPass<spirv::ModuleOp>(
        spirv::createSPIRVLowerABIAttributesPass());
    pm.addNestedPass<spirv::ModuleOp>(spirv::createSPIRVUpdateVCEPass());

    // Create Vulkan dispatch.
    pm.addPass(createConvertGpuLaunchFuncToVulkanLaunchFuncPass());
    pm.addNestedPass<func::FuncOp>(LLVM::createRequestCWrappersPass());

    // Cleanup IR.
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
#endif // TPP_GPU_ENABLE
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::tpp::createGpuToVulkanPass() {
  return std::make_unique<GpuToVulkan>();
}
