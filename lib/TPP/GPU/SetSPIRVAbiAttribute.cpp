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
    auto attrName = StringAttr::get(context, spirv::getEntryPointABIAttrName());
    if (clientAPI == "opencl") {
      auto abi = spirv::getEntryPointABIAttr(context);
      for (const auto &gpuFunc : gpuModule.getOps<gpu::GPUFuncOp>()) {
        if (!gpu::GPUDialect::isKernel(gpuFunc) || gpuFunc->getAttr(attrName))
          continue;

        gpuFunc->setAttr(attrName, abi);
      }
    } else if (clientAPI == "vulkan") {
      const SmallVector<gpu::Dimension> dims = {
          gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z};

      for (Operation *gpuFunc : gpuModule.getOps<gpu::GPUFuncOp>()) {
        if (!gpu::GPUDialect::isKernel(gpuFunc) || gpuFunc->getAttr(attrName))
          continue;

        SmallVector<int32_t> dimSizes;
        for (auto &dim : dims) {
          auto blockSize = cast<gpu::GPUFuncOp>(gpuFunc).getKnownBlockSize(dim);
          uint32_t size = blockSize ? *blockSize : 1;
          dimSizes.push_back(size);
        }
        auto abi = spirv::getEntryPointABIAttr(context, dimSizes);

        gpuFunc->setAttr(attrName, abi);
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
mlir::tpp::createSetSPIRVAbiAttributePass(StringRef api) {
  return std::make_unique<SetSPIRVAbiAttributePass>(api);
}