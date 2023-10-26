//===- GpuVulkanAbi.cpp ------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include <numeric>

using namespace mlir;
using namespace mlir::tpp;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_GPUVULKANABI
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

// Flatten memref type into 1D shape.
static Type FlattenMemrefType(MemRefType memrefType) {
  if (memrefType.getRank() <= 1)
    return memrefType;

  // If the shape is not static, replace memref with a fully dynamic 1D shape.
  // Otherwise, compute new flat size.
  int64_t flatSize = ShapedType::kDynamic;
  if (memrefType.hasStaticShape()) {
    auto shape = memrefType.getShape();
    flatSize = std::accumulate(shape.begin(), shape.end(), 1,
                               std::multiplies<int64_t>());
  }
  auto flatMemref = MemRefType::get({flatSize}, memrefType.getElementType());

  return flatMemref;
}

// Returns Vulkan compatible wrapper type for a given input type.
static Type getVulkanTypeWrapper(Type type,
                                 const SPIRVTypeConverter &typeConverter,
                                 RewriterBase &rewriter) {
  assert(!type.isa<TensorType>() && "Tensors are not supported by Vulkan");

  // Buffers are already Vulkan compatible.
  if (auto memrefType = type.dyn_cast<MemRefType>())
    return FlattenMemrefType(memrefType);

  // Index has to be converted to a fixed-size integer.
  if (type.isa<IndexType>()) {
    auto spirvIndex = typeConverter.getIndexType().cast<spirv::SPIRVType>();
    type = rewriter.getIntegerType(*(spirvIndex.getSizeInBytes()) * 8);
  }

  // Wrap the basic type in a buffer.
  auto typeWrapper = MemRefType::get({1}, type);
  return typeWrapper;
};

// Rewrites GPU function to have Vulkan compatible signature.
// Replaces the original kernel with an adapted empty kernel.
static void adaptGPUFuncToVulkanABI(gpu::GPUFuncOp &gpuFuncOp,
                                    const SPIRVTypeConverter &typeConverter,
                                    RewriterBase &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(gpuFuncOp.getOperation());

  // Create a new empty GPU kernel.
  // The original logic is already captured by earlier conversion from GPU to
  // SPIRV modules and an empty body ensures that kernel inputs can be easily
  // modified.
  auto funcOp = rewriter.cloneWithoutRegions(gpuFuncOp);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&funcOp.getRegion().emplaceBlock());
    rewriter.create<gpu::ReturnOp>(gpuFuncOp.getLoc());
  }

  // Buffers are Vulkan compatible but need to be min 1D and max 3D shaped.
  // Scalars cannot be passed directly to a Vulkan kernels.
  // Wrap them into Vulkan compatible buffers.
  auto funcType = funcOp.getFunctionType();
  SmallVector<Type> inputs;
  for (Type input : funcType.getInputs())
    inputs.push_back(getVulkanTypeWrapper(input, typeConverter, rewriter));

  // Change the function input types.
  auto newFuncType = funcType.clone(inputs, funcType.getResults());
  funcOp.setFunctionType(newFuncType);

  // Add block arguments that correspond to the new argument types.
  auto &block = funcOp.getRegion().front();
  for (auto type : inputs) {
    block.addArgument(type, funcOp.getLoc());
  }

  // Remove the original kernel.
  rewriter.eraseOp(gpuFuncOp.getOperation());
}

// Flatten memref buffer into 1D shape.
static Value FlattenMemrefOperand(Value operand, RewriterBase &rewriter) {
  auto loc = operand.getLoc();

  // Ignore non-memref types and 1D buffers.
  auto memrefType = operand.getType().dyn_cast<MemRefType>();
  if (!memrefType || memrefType.getRank() <= 1)
    return operand;

  // Collapse all dimensions into a 1D shape.
  ReassociationIndices reassociation;
  for (unsigned i = 0; i < memrefType.getShape().size(); i++)
    reassociation.push_back(i);

  auto collapseOp = rewriter.create<memref::CollapseShapeOp>(
      loc, operand, SmallVector<ReassociationIndices>{reassociation});

  return collapseOp.getResult();
}

// Returns Vulkan compatible wrapper value for a given input operand.
static Value getVulkanOperandWrapper(Value operand,
                                     const SPIRVTypeConverter &typeConverter,
                                     RewriterBase &rewriter) {
  auto loc = operand.getLoc();
  auto type = operand.getType();

  assert(!type.isa<TensorType>() && "Tensors are not supported by Vulkan");

  // Buffers are Vulkan compatible but need to be min 1D and max 3D shaped.
  if (type.isa<MemRefType>())
    return FlattenMemrefOperand(operand, rewriter);

  auto wrapperType =
      getVulkanTypeWrapper(type, typeConverter, rewriter).cast<MemRefType>();

  // Cast index to a fixed-size integer.
  if (type.isa<IndexType>()) {
    auto castType = wrapperType.getElementType();
    operand =
        rewriter.create<arith::IndexCastOp>(loc, castType, operand).getOut();
  }

  // Scalar values are small so put them on stack.
  auto wrapper = rewriter.create<memref::AllocaOp>(loc, wrapperType);
  auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  rewriter.create<memref::StoreOp>(loc, operand, wrapper, ValueRange{zero});

  return wrapper;
}

// Adapts GPU function launch to Vulkan calling convention.
// Scalar values are wrapped into Vulkan compatible buffers.
static void
adaptGPULaunchFuncToVulkanABI(gpu::LaunchFuncOp &gpuLaunchFuncOp,
                              const SPIRVTypeConverter &typeConverter,
                              RewriterBase &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(gpuLaunchFuncOp.getOperation());

  // Buffers are Vulkan compatible but need to be min 1D and max 3D shaped.
  // Scalars cannot be passed directly to a Vulkan kernels.
  // Wrap them into Vulkan compatible buffers.
  SmallVector<Value> newOperands;
  for (Value operand : gpuLaunchFuncOp.getKernelOperands()) {
    newOperands.push_back(
        getVulkanOperandWrapper(operand, typeConverter, rewriter));
  }

  // Update function launch arguments.
  gpuLaunchFuncOp.getKernelOperandsMutable().assign(newOperands);
}

struct GpuVulkanAbi : public tpp::impl::GpuVulkanAbiBase<GpuVulkanAbi> {
  using GpuVulkanAbiBase::GpuVulkanAbiBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect>();
    registry.insert<spirv::SPIRVDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    IRRewriter rewriter(&getContext());

    SPIRVConversionOptions options;
    options.use64bitIndex = this->use64bitIndex;

    SmallVector<gpu::GPUFuncOp, 1> gpuFuncs;
    module.walk([&](gpu::GPUFuncOp gpuFuncOp) {
      // Only gather GPU kernel functions.
      if (gpuFuncOp.isKernel())
        gpuFuncs.push_back(gpuFuncOp);
    });
    for (auto &gpuFunc : gpuFuncs) {
      auto targetAttr = spirv::lookupTargetEnvOrDefault(gpuFunc);
      std::unique_ptr<ConversionTarget> target =
          SPIRVConversionTarget::get(targetAttr);
      SPIRVTypeConverter typeConverter(targetAttr, options);

      adaptGPUFuncToVulkanABI(gpuFunc, typeConverter, rewriter);
    }

    SmallVector<gpu::LaunchFuncOp, 1> gpuLaunches;
    module.walk([&](gpu::LaunchFuncOp gpuLaunchOp) {
      gpuLaunches.push_back(gpuLaunchOp);
    });
    for (auto &launchFuncOp : gpuLaunches) {
      auto targetAttr =
          spirv::lookupTargetEnvOrDefault(launchFuncOp.getOperation());
      std::unique_ptr<ConversionTarget> target =
          SPIRVConversionTarget::get(targetAttr);
      SPIRVTypeConverter typeConverter(targetAttr, options);

      adaptGPULaunchFuncToVulkanABI(launchFuncOp, typeConverter, rewriter);
    }
  }
};

} // namespace
