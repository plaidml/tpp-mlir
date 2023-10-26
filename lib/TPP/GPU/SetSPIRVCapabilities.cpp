//===- SetSPIRVCapabilities.cpp ----------------------------------*- C++-*-===//
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

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_SETSPIRVCAPABILITIES
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

struct SetSPIRVCapabilities
    : public tpp::impl::SetSPIRVCapabilitiesBase<SetSPIRVCapabilities> {
  using SetSPIRVCapabilitiesBase::SetSPIRVCapabilitiesBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect>();
    registry.insert<spirv::SPIRVDialect>();
  }

  void runOnOperation() override {
    namespace spirv = mlir::spirv;
    auto context = &getContext();
    spirv::Capability caps_opencl[] = {
        // clang-format off
        spirv::Capability::Addresses,
        spirv::Capability::Float16Buffer,
        spirv::Capability::Int64,
        spirv::Capability::Int16,
        spirv::Capability::Int8,
        spirv::Capability::Bfloat16ConversionINTEL,
        spirv::Capability::Kernel,
        spirv::Capability::Linkage,
        spirv::Capability::Vector16,
        spirv::Capability::GenericPointer,
        spirv::Capability::Groups,
        spirv::Capability::Float16,
        spirv::Capability::Float64,
        spirv::Capability::AtomicFloat32AddEXT,
        spirv::Capability::ExpectAssumeKHR,
        spirv::Capability::CooperativeMatrixNV,
        spirv::Capability::StorageBuffer16BitAccess,
        // clang-format on
    };
    spirv::Capability caps_vulkan[] = {
        // clang-format off
        spirv::Capability::Shader,
        spirv::Capability::CooperativeMatrixNV,
        spirv::Capability::Float16,
        spirv::Capability::StorageBuffer16BitAccess,
        // clang-format on
    };
    spirv::Extension exts_opencl[] = {
        spirv::Extension::SPV_INTEL_bfloat16_conversion,
        spirv::Extension::SPV_EXT_shader_atomic_float_add,
        spirv::Extension::SPV_KHR_expect_assume,
        spirv::Extension::SPV_KHR_16bit_storage,
        spirv::Extension::SPV_NV_cooperative_matrix};
    spirv::Extension exts_vulkan[] = {
        spirv::Extension::SPV_KHR_storage_buffer_storage_class,
        spirv::Extension::SPV_KHR_16bit_storage,
        spirv::Extension::SPV_NV_cooperative_matrix};
    if (clientAPI == "opencl") {
      auto triple = spirv::VerCapExtAttr::get(
          spirv::Version::V_1_4, caps_opencl, exts_opencl, context);
      auto attr = spirv::TargetEnvAttr::get(
          triple, spirv::getDefaultResourceLimits(context),
          spirv::ClientAPI::OpenCL, spirv::Vendor::Unknown,
          spirv::DeviceType::Unknown, spirv::TargetEnvAttr::kUnknownDeviceID);
      auto op = getOperation();
      op->walk([&](mlir::gpu::GPUModuleOp op) {
        op->setAttr(spirv::getTargetEnvAttrName(), attr);
      });
    } else if (clientAPI == "vulkan") {
      auto triple = spirv::VerCapExtAttr::get(
          spirv::Version::V_1_4, caps_vulkan, exts_vulkan, context);
      auto attr = spirv::TargetEnvAttr::get(
          triple, spirv::getDefaultResourceLimits(context),
          spirv::ClientAPI::Vulkan, spirv::Vendor::Unknown,
          spirv::DeviceType::Unknown, spirv::TargetEnvAttr::kUnknownDeviceID);
      auto op = getOperation();
      op->walk([&](mlir::gpu::GPUModuleOp op) {
        op->setAttr(spirv::getTargetEnvAttrName(), attr);
      });
    }
  }
};

} // namespace
