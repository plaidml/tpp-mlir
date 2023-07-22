//===- SetSPIRVAbiAttribute.cpp ----------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {
class SetSPIRVAbiAttributePass
    : public SetSPIRVAbiAttributeBase<SetSPIRVAbiAttributePass> {
public:
  SetSPIRVAbiAttributePass() = default;
  SetSPIRVAbiAttributePass(StringRef clientAPI) {
    this->clientAPI = clientAPI.str();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect>();
    registry.insert<spirv::SPIRVDialect>();
  }

  void runOnOperation() override {
    auto gpuModule = getOperation();
    auto *context = &getContext();
    auto attrName =
        mlir::StringAttr::get(context, mlir::spirv::getEntryPointABIAttrName());
    if (clientAPI == "opencl") {
      auto abi = mlir::spirv::getEntryPointABIAttr(context);
      for (const auto &gpuFunc : gpuModule.getOps<mlir::gpu::GPUFuncOp>()) {
        if (!mlir::gpu::GPUDialect::isKernel(gpuFunc) ||
            gpuFunc->getAttr(attrName))
          continue;

        gpuFunc->setAttr(attrName, abi);
      }
    } else if (clientAPI == "vulkan") {
      auto abi = mlir::spirv::getEntryPointABIAttr(context, {1, 1, 1});
      for (const auto &gpuFunc : gpuModule.getOps<mlir::gpu::GPUFuncOp>()) {
        if (!mlir::gpu::GPUDialect::isKernel(gpuFunc) ||
            gpuFunc->getAttr(attrName))
          continue;

        gpuFunc->setAttr(attrName, abi);
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::tpp::createSetSPIRVAbiAttributePass(StringRef api) {
  return std::make_unique<SetSPIRVAbiAttributePass>(api);
}