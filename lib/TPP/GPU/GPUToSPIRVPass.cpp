//===- GPUToSPIRVPass.cpp - GPU to SPIR-V Passes --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert a kernel function in the GPU Dialect
// into a spirv.module operation.
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"

#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"

#include "mlir/Conversion/ArithToSPIRV/ArithToSPIRV.h"
#include "mlir/Conversion/FuncToSPIRV/FuncToSPIRV.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h"
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRV.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include <mlir/Conversion/ArithToSPIRV/ArithToSPIRV.h>
#include <mlir/Conversion/ControlFlowToSPIRV/ControlFlowToSPIRV.h>
#include <mlir/Conversion/FuncToSPIRV/FuncToSPIRV.h>
#include <mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h>
#include <mlir/Conversion/MathToSPIRV/MathToSPIRV.h>
#include <mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h>

using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {
/// Pass to lower GPU Dialect to SPIR-V. The pass only converts the gpu.func ops
/// inside gpu.module ops. i.e., the function that are referenced in
/// gpu.launch_func ops. For each such function
///
/// 1) Create a spirv::ModuleOp, and clone the function into spirv::ModuleOp
/// (the original function is still needed by the gpu::LaunchKernelOp, so cannot
/// replace it).
///
/// 2) Lower the body of the spirv::ModuleOp.
class GPUToSPIRVPass : public GPUToSPIRVBase<GPUToSPIRVPass> {
public:
  explicit GPUToSPIRVPass(bool mapMemorySpace)
      : mapMemorySpace(mapMemorySpace) {}
  void runOnOperation() override;

private:
  bool mapMemorySpace;
};
} // namespace

static Type getSPIRVTypeWrapper(Type type,
                                const SPIRVTypeConverter &typeConverter,
                                RewriterBase &rewriter) {
  // Buffers are already SPIRV compatible
  if (type.isa<ShapedType>())
    return type;

  // Index has to be converted to a fixed-size integer
  if (type.isa<IndexType>()) {
    auto spirvIndex = typeConverter.getIndexType().cast<spirv::SPIRVType>();
    type = rewriter.getIntegerType(*(spirvIndex.getSizeInBytes()) * 8);
  }

  // Wrap the basic type in a buffer
  auto typeWrapper = MemRefType::get({1}, type);
  return typeWrapper;
};

static void adaptGPUFuncToSPIRVABI(gpu::GPUFuncOp &gpuFuncOp,
                                   const SPIRVTypeConverter &typeConverter,
                                   RewriterBase &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(gpuFuncOp.getOperation());
  auto funcOp = rewriter.cloneWithoutRegions(gpuFuncOp);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&funcOp.getRegion().emplaceBlock());
    rewriter.create<gpu::ReturnOp>(gpuFuncOp.getLoc());
  }

  auto funcType = funcOp.getFunctionType();
  SmallVector<Type> inputs;
  for (Type input : funcType.getInputs()) {
    if (input.isa<ShapedType>())
      inputs.push_back(input);
    else
      inputs.push_back(getSPIRVTypeWrapper(input, typeConverter, rewriter));
  }
  auto newFuncType = funcType.clone(inputs, funcType.getResults());
  funcOp.setFunctionType(newFuncType);

  auto &block = funcOp.getRegion().front();
  for (auto type : inputs) {
    block.addArgument(type, funcOp.getLoc());
  }

  rewriter.eraseOp(gpuFuncOp.getOperation());
}

static void
adaptGPULaunchFuncToSPIRVABI(gpu::LaunchFuncOp &gpuLaunchFuncOp,
                             const SPIRVTypeConverter &typeConverter,
                             RewriterBase &rewriter) {
  auto loc = gpuLaunchFuncOp.getLoc();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(gpuLaunchFuncOp.getOperation());

  SmallVector<Value> newOperands;
  for (Value operand : gpuLaunchFuncOp.getKernelOperands()) {
    auto type = operand.getType();
    if (type.isa<ShapedType>()) {
      newOperands.push_back(operand);
      continue;
    }

    auto wrapperType =
        getSPIRVTypeWrapper(type, typeConverter, rewriter).cast<MemRefType>();
    if (type.isa<IndexType>()) {
      auto castType = wrapperType.getElementType();
      operand =
          rewriter.create<arith::IndexCastOp>(loc, castType, operand).getOut();
    }
    auto wrapper = rewriter.create<memref::AllocaOp>(loc, wrapperType);
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    rewriter.create<memref::StoreOp>(loc, operand, wrapper, ValueRange{zero});
    newOperands.push_back(wrapper);
  }

  gpuLaunchFuncOp.getKernelOperandsMutable().assign(newOperands);
}

void GPUToSPIRVPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();
  IRRewriter rewriter(context);

  SPIRVConversionOptions options;
  options.use64bitIndex = this->use64bitIndex;

  SmallVector<Operation *, 1> gpuModules;
  module.walk([&](gpu::GPUModuleOp moduleOp) {
    // Clone each GPU kernel module for conversion, given that the GPU
    // launch op still needs the original GPU kernel module.
    rewriter.setInsertionPoint(moduleOp.getOperation());
    gpuModules.push_back(rewriter.clone(*moduleOp.getOperation()));

    SmallVector<gpu::GPUFuncOp, 1> gpuFuncs;
    moduleOp.walk([&](gpu::GPUFuncOp gpuFuncOp) {
      // Only gather GPU kernel functions.
      if (gpuFuncOp.isKernel())
        gpuFuncs.push_back(gpuFuncOp);
    });
    for (auto &gpuFuncOp : gpuFuncs) {
      auto targetAttr = spirv::lookupTargetEnvOrDefault(gpuFuncOp);
      std::unique_ptr<ConversionTarget> target =
          SPIRVConversionTarget::get(targetAttr);
      SPIRVTypeConverter typeConverter(targetAttr, options);

      adaptGPUFuncToSPIRVABI(gpuFuncOp, typeConverter, rewriter);
    }
  });

  SmallVector<gpu::LaunchFuncOp, 1> gpuLaunches;
  module.walk([&](gpu::LaunchFuncOp gpuLaunchOp) {
    gpuLaunches.push_back(gpuLaunchOp);
  });
  for (auto &launchOp : gpuLaunches) {
    auto targetAttr = spirv::lookupTargetEnvOrDefault(launchOp.getOperation());
    std::unique_ptr<ConversionTarget> target =
        SPIRVConversionTarget::get(targetAttr);
    SPIRVTypeConverter typeConverter(targetAttr, options);

    adaptGPULaunchFuncToSPIRVABI(launchOp, typeConverter, rewriter);
  }

  // Run conversion for each module independently as they can have different
  // TargetEnv attributes.
  for (Operation *gpuModule : gpuModules) {
    // Map MemRef memory space to SPIR-V storage class first if requested.
    if (mapMemorySpace) {
      std::unique_ptr<ConversionTarget> target =
          spirv::getMemorySpaceToStorageClassTarget(*context);
      spirv::MemorySpaceToStorageClassMap memorySpaceMap =
          spirv::mapMemorySpaceToVulkanStorageClass;
      spirv::MemorySpaceToStorageClassConverter converter(memorySpaceMap);

      RewritePatternSet patterns(context);
      spirv::populateMemorySpaceToStorageClassPatterns(converter, patterns);

      if (failed(applyFullConversion(gpuModule, *target, std::move(patterns))))
        return signalPassFailure();
    }

    auto targetAttr = spirv::lookupTargetEnvOrDefault(gpuModule);
    std::unique_ptr<ConversionTarget> target =
        SPIRVConversionTarget::get(targetAttr);

    SPIRVTypeConverter typeConverter(targetAttr, options);
    typeConverter.addConversion([&](gpu::MMAMatrixType type) -> Type {
      return convertMMAToSPIRVType(type);
    });
    RewritePatternSet patterns(context);
    populateGPUToSPIRVPatterns(typeConverter, patterns);
    populateGpuWMMAToSPIRVConversionPatterns(typeConverter, patterns);
    // TODO: Change SPIR-V conversion to be progressive and remove the following
    // patterns.
    mlir::arith::populateArithToSPIRVPatterns(typeConverter, patterns);
    populateMemRefToSPIRVPatterns(typeConverter, patterns);
    populateFuncToSPIRVPatterns(typeConverter, patterns);

    // TODO: upstream the extra pattern registration if they work well
    mlir::ScfToSPIRVContext scfToSpirvCtx;
    mlir::populateSCFToSPIRVPatterns(typeConverter, scfToSpirvCtx, patterns);
    mlir::cf::populateControlFlowToSPIRVPatterns(typeConverter, patterns);
    mlir::populateMathToSPIRVPatterns(typeConverter, patterns);

    if (failed(applyFullConversion(gpuModule, *target, std::move(patterns))))
      return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::tpp::createConvertGPUToSPIRVPass(bool mapMemorySpace) {
  return std::make_unique<GPUToSPIRVPass>(mapMemorySpace);
}
