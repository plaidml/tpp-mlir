//===- DefaultTppPasses.cpp --------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "TPP/Dialect/Check/BufferizableOpInterfaceImpl.h"
#include "TPP/Dialect/Check/CheckDialect.h"
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
  DefaultTppPasses() = default;
  DefaultTppPasses(bool tppToLoops) { this->tppToLoops = tppToLoops; };

  void getDependentDialects(DialectRegistry &registry) const override {
    // Add all custom TPP dialects.
    registry.insert<tpp::TppDialect>();
    registry.insert<xsmm::XsmmDialect>();
    registry.insert<check::CheckDialect>();
    registry.insert<vnni::VNNIDialect>();
    registry.insert<perf::PerfDialect>();
    bufferization::registerAllocationOpInterfaceExternalModels(registry);
    linalgx::registerTransformDialectExtension(registry);
    check::registerBufferizableOpInterfaceExternalModels(registry);
    vnni::registerBufferizableOpInterfaceExternalModels(registry);
    perf::registerBufferizableOpInterfaceExternalModels(registry);

    // Add all core MLIR dialects as the default TPP passes may contain any
    // combination of other passes.
    registerAllDialects(registry);
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    PassManager pm(module.getContext(), mlir::OpPassManager::Nesting::Implicit);

    // Run transforms first and clean them up afterwards.
    pm.addPass(createTransformDialectInterpreterPass());
    pm.addPass(createTransformDropSchedulePass());

    // Preprocess convolutions.
    pm.addNestedPass<func::FuncOp>(createDecomposeConvToMatmulOrBrgemmPass());

    // Materialize empty tensors
    pm.addPass(bufferization::createEmptyTensorToAllocTensorPass());

    // Run bufferization as the rest of the passes prefer working on memref.
    bufferization::OneShotBufferizationOptions buffOpts;
    buffOpts.allowReturnAllocs = true;
    buffOpts.bufferizeFunctionBoundaries = true;
    buffOpts.functionBoundaryTypeConversion =
        bufferization::LayoutMapOption::IdentityLayoutMap;
    pm.addPass(bufferization::createOneShotBufferizePass(buffOpts));
    pm.addPass(bufferization::createDropEquivalentBufferResultsPass());
    pm.addNestedPass<func::FuncOp>(
        bufferization::createFinalizingBufferizePass());
    // Clean up after bufferization.
    pm.addPass(bufferization::createBufferDeallocationPass());
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());

    // Convert generics to BRGEMM.
    // The mapping is done after bufferization as the buffer semantics
    // allow direct use of scf.parallel loops. This prevents different
    // lowering outputs between input linalg on tensors and memrefs.
    pm.addNestedPass<func::FuncOp>(createMapToBatchReduceGEMMPass());

    // Convert all higher level dialects to TPP.
    pm.addNestedPass<func::FuncOp>(createConvertLinalgToTppPass());
    pm.addPass(createConvertVNNIToTppPass());

    // Lower all TPP ops.
    if (tppToLoops)
      pm.addNestedPass<func::FuncOp>(createConvertTppToLoopsPass());
    else
      pm.addNestedPass<func::FuncOp>(createConvertTppToXsmmPass());

    // Lower all Check ops.
    pm.addPass(createConvertCheckToLoopsPass());

    // Postprocess generated loops.
    // Perform LICM before function calls are generated to ensure that ops which
    // map directly to functions also get moved outside of loops, if possible.
    // This approach assumes that the function calls do not have any side
    // effects and can be safely moved outside of loop body.
    pm.addPass(createLoopInvariantCodeMotionPass());
    pm.addPass(createParallelLoopFusionPass());

    // Lower all XSMM ops.
    pm.addPass(createConvertXsmmToFuncPass());

    // General postprocessing.
    pm.addPass(bufferization::createBufferHoistingPass());
    pm.addPass(bufferization::createBufferDeallocationPass());
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    if (failed(runPipeline(pm, module)))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::tpp::createDefaultTppPass() {
  return std::make_unique<DefaultTppPasses>();
}
