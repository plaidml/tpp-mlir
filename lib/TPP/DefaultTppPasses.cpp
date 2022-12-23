//===- DefaultTppPasses.cpp --------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"

#include "TPP/Dialect/Check/BufferizableOpInterfaceImpl.h"
#include "TPP/Dialect/Check/CheckDialect.h"
#include "TPP/Dialect/LinalgX/BufferizableOpInterfaceImpl.h"
#include "TPP/Dialect/LinalgX/LinalgXDialect.h"
#include "TPP/Dialect/Perf/BufferizableOpInterfaceImpl.h"
#include "TPP/Dialect/Perf/PerfDialect.h"
#include "TPP/Dialect/Tpp/TppDialect.h"
#include "TPP/Dialect/Transform/LinalgXTransformOps.h"
#include "TPP/Dialect/VNNI/BufferizableOpInterfaceImpl.h"
#include "TPP/Dialect/VNNI/VNNIDialect.h"
#include "TPP/Dialect/Xsmm/XsmmDialect.h"

using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

struct DefaultTppPasses : public DefaultTppPassesBase<DefaultTppPasses> {
  void getDependentDialects(DialectRegistry &registry) const override {
    // Add all custom TPP dialects
    registry.insert<tpp::TppDialect>();
    registry.insert<xsmm::XsmmDialect>();
    registry.insert<linalgx::LinalgXDialect>();
    registry.insert<check::CheckDialect>();
    registry.insert<vnni::VNNIDialect>();
    registry.insert<perf::PerfDialect>();
    bufferization::registerAllocationOpInterfaceExternalModels(registry);
    linalgx::registerTransformDialectExtension(registry);
    linalgx::registerBufferizableOpInterfaceExternalModels(registry);
    check::registerBufferizableOpInterfaceExternalModels(registry);
    vnni::registerBufferizableOpInterfaceExternalModels(registry);
    perf::registerBufferizableOpInterfaceExternalModels(registry);

    // Add all core MLIR dialects as the default TPP passes may contain any
    // combination of the existing passes.
    registerAllDialects(registry);
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    PassManager pm(module.getContext(), mlir::OpPassManager::Nesting::Implicit);

    pm.addNestedPass<func::FuncOp>(createMapLinalgToTppPass());

    bufferization::OneShotBufferizationOptions buffOpts;
    buffOpts.allowReturnAllocs = true;
    buffOpts.bufferizeFunctionBoundaries = true;
    buffOpts.functionBoundaryTypeConversion =
        bufferization::LayoutMapOption::IdentityLayoutMap;
    pm.addPass(bufferization::createOneShotBufferizePass(buffOpts));
    pm.addPass(bufferization::createDropEquivalentBufferResultsPass());
    pm.addNestedPass<func::FuncOp>(
        bufferization::createFinalizingBufferizePass());
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());

    pm.addNestedPass<func::FuncOp>(createConvertLinalgToTppPass());
    pm.addNestedPass<func::FuncOp>(createConvertTppToXsmmPass());

    if (failed(runPipeline(pm, module)))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::tpp::createDefaultTppPass() {
  return std::make_unique<DefaultTppPasses>();
}
