//===- GpuToVulkan.cpp -------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/RequestCWrappers.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
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

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_GPUTOVULKAN
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

// Lower generic GPU ops to Vulkan backend.
struct GpuToVulkan : public tpp::impl::GpuToVulkanBase<GpuToVulkan>,
                     UtilityPassBase<ModuleOp> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect>();
    registry.insert<spirv::SPIRVDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<func::FuncDialect>();
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

#ifdef TPP_VULKAN_ENABLE
    // Preprocess
    // Subviews are not supported by SPIRV ops
    pm.addPass(memref::createFoldMemRefAliasOpsPass());
    pm.addPass(createLowerAffinePass());

    // Create SPIRV kernels.
    pm.addPass(tpp::createSetSPIRVCapabilitiesPass());
    pm.addPass(tpp::createSetSPIRVAbiAttributePass());
    pm.addPass(tpp::createConvertGPUToSPIRVPass());
    pm.addNestedPass<spirv::ModuleOp>(
        spirv::createSPIRVLowerABIAttributesPass());
    pm.addNestedPass<spirv::ModuleOp>(spirv::createSPIRVUpdateVCEPass());

    // Adapt GPU kernel to be compliant with Vulkan ABI.
    pm.addPass(tpp::createGpuVulkanAbiPass());

    // Create Vulkan dispatch.
    pm.addPass(createConvertGpuLaunchFuncToVulkanLaunchFuncPass());
    pm.addNestedPass<func::FuncOp>(LLVM::createRequestCWrappersPass());

    // Cleanup IR.
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
#endif // TPP_VULKAN_ENABLE
  }
};

} // namespace
