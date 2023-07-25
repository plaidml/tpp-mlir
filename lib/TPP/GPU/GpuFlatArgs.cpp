//===- GpuFlatArgsPass.cpp ---------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#include <numeric>

using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

#define DEBUG_TYPE "gpu-flat-args"

namespace {

void FlattenArgsGpuFunc(gpu::GPUFuncOp gpuFunc, RewriterBase &rewriter) {
  auto &body = gpuFunc.getBody();
  const auto numOps = std::distance(body.front().begin(), body.front().end());
  if (numOps > 1) {
    LLVM_DEBUG(llvm::dbgs() << "Cannot flatten args of non-empty GPU kernel\n");
    return;
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(gpuFunc.getOperation());

  // Create a new empty GPU kernel.
  auto funcOp = rewriter.cloneWithoutRegions(gpuFunc);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&funcOp.getRegion().emplaceBlock());
    rewriter.create<gpu::ReturnOp>(gpuFunc.getLoc());
  }

  auto funcType = funcOp.getFunctionType();
  SmallVector<Type> inputs;
  for (Type input : funcType.getInputs()) {
    auto memrefType = input.dyn_cast<MemRefType>();

    if (!memrefType) {
      inputs.push_back(input);
      continue;
    }

    int64_t flatSize = ShapedType::kDynamic;
    if (memrefType.hasStaticShape()) {
      auto shape = memrefType.getShape();
      flatSize = std::accumulate(shape.begin(), shape.end(), 1,
                                 std::multiplies<int64_t>());
    }

    auto flatType = MemRefType::get({flatSize}, memrefType.getElementType());
    inputs.push_back(flatType);
  }

  auto newFuncType = funcType.clone(inputs, funcType.getResults());
  funcOp.setFunctionType(newFuncType);

  // Add block arguments that correspond to the new argument types.
  auto &block = funcOp.getBody().front();
  for (auto type : inputs) {
    block.addArgument(type, funcOp.getLoc());
  }

  // Remove the original kernel.
  rewriter.eraseOp(gpuFunc.getOperation());
}

void FlattenArgsGpuLaunchFunc(gpu::LaunchFuncOp launchFuncOp,
                              RewriterBase &rewriter) {
  return;
}

class GpuFlatArgs : public GpuFlatArgsBase<GpuFlatArgs> {
public:
  GpuFlatArgs() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect>();
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    IRRewriter rewriter(&getContext());

    SmallVector<gpu::GPUFuncOp, 1> gpuFuncs;
    module.walk(
        [&](gpu::GPUFuncOp gpuFuncOp) { gpuFuncs.push_back(gpuFuncOp); });
    for (auto &gpuFunc : gpuFuncs) {
      FlattenArgsGpuFunc(gpuFunc, rewriter);
    }

    SmallVector<gpu::LaunchFuncOp, 1> gpuLaunches;
    module.walk([&](gpu::LaunchFuncOp gpuLaunchOp) {
      gpuLaunches.push_back(gpuLaunchOp);
    });
    for (auto &launchFuncOp : gpuLaunches) {
      FlattenArgsGpuLaunchFunc(launchFuncOp, rewriter);
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::tpp::createGpuFlatArgsPass() {
  return std::make_unique<GpuFlatArgs>();
}
