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

// Prototypes
std::unique_ptr<OperationPass<ModuleOp>> createCleanupPass();
std::unique_ptr<OperationPass<ModuleOp>> createTransformPass();
std::unique_ptr<OperationPass<ModuleOp>> createBufferizationPass();
std::unique_ptr<OperationPass<ModuleOp>> createLocalDialectsLoweringPass();
std::unique_ptr<OperationPass<ModuleOp>> createPostprocessingPass();

template <typename DerivedT, typename OpT = ModuleOp>
class UtilityPassBase : public PassWrapper<DerivedT, OperationPass<OpT>> {
protected:
  OpPassManager pm;

  // Create the pass processing pipeline.
  virtual void constructPipeline() = 0;
};

// A general cleanup pass that performs general IR normalization and
// generic optimizations without any lowering or any logical changes.
// Commonly applied after other major passes.
struct CleanupPass : public UtilityPassBase<CleanupPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CleanupPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();

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

    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createCleanupPass() {
  return std::make_unique<CleanupPass>();
}

// Apply any present transforms and remove transform blocks afterwards.
struct TransformPass : public UtilityPassBase<TransformPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TransformPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();

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

    // Run all transforms and clean them up afterwards.
    pm.addPass(createTransformDialectInterpreterPass());
    pm.addPass(createTransformDropSchedulePass());
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createTransformPass() {
  return std::make_unique<TransformPass>();
}

// Apply global bufferization - convert all tensors to memrefs.
// Uses TPP-specific bufferization options.
struct BufferizationPass : public UtilityPassBase<BufferizationPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BufferizationPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();

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
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createBufferizationPass() {
  return std::make_unique<BufferizationPass>();
}

// Lower all local dialects (XSMM, check etc.) to standard dialects
// and function calls.
struct LocalDialectsLoweringPass
    : public UtilityPassBase<LocalDialectsLoweringPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LocalDialectsLoweringPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();

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

    pm.addPass(createConvertCheckToLoopsPass());
    pm.addPass(createConvertXsmmToFuncPass());
    pm.addPass(createConvertPerfToLoopsPass());
    pm.addPass(createConvertPerfToFuncPass());
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createLocalDialectsLoweringPass() {
  return std::make_unique<LocalDialectsLoweringPass>();
}

// Apply various postprocessing passes such as LICM, parallel loop fusion,
// buffer deallocation, general cleanup etc.
struct PostprocessingPass : public UtilityPassBase<PostprocessingPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PostprocessingPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();

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

    // Postprocess generated loops.
    pm.addPass(createLoopInvariantCodeMotionPass());
    pm.addPass(createRaiseToParallelLoopPass());
    pm.addPass(createParallelLoopFusionPass());

    // Postprocess buffers.
    pm.addPass(bufferization::createBufferHoistingPass());
    pm.addPass(bufferization::createBufferDeallocationPass());

    // Run general cleanup to normalize IR.
    pm.addPass(createCleanupPass());
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createPostprocessingPass() {
  return std::make_unique<PostprocessingPass>();
}

struct DefaultTppPasses : public DefaultTppPassesBase<DefaultTppPasses> {
  DefaultTppPasses() : DefaultTppPasses(false, false){};
  DefaultTppPasses(bool tppToLoops, bool linalgToLoops)
      : pm("builtin.module", mlir::OpPassManager::Nesting::Implicit) {
    this->tppToLoops = tppToLoops;
    this->linalgToLoops = linalgToLoops;
  };

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

    // Initialize the pipeline if needed.
    // Otherwise, just run the cached one.
    if (pm.empty())
      constructPipeline();

    if (failed(runPipeline(pm, module)))
      return signalPassFailure();
  }

private:
  OpPassManager pm;

  // Create the default processing pipeline.
  void constructPipeline() {
    pm.clear();

    // Run transforms first and clean them up afterwards.
    pm.addPass(createTransformDialectInterpreterPass());
    pm.addPass(createTransformDropSchedulePass());

    if (linalgToLoops) {
      // Lower linalg directly to loops.
      // Skip all TPP transformations.
      pm.addPass(createBufferizePass());
      pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
    } else {
      // Preprocess convolutions.
      pm.addNestedPass<func::FuncOp>(createRewriteConvToMatmulOrBrgemmPass());

      // Generalize tensor.pack and tensor.unpack.
      pm.addNestedPass<func::FuncOp>(createGeneralizeTensorPackAndUnPackPass());

      // Run bufferization as the rest of the passes prefer working on memref.
      pm.addPass(createBufferizePass());

      // Convert generics to BRGEMM.
      // The mapping is done after bufferization as the buffer semantics
      // allow direct use of scf.parallel loops. This prevents different
      // lowering outputs between input linalg on tensors and memrefs.
      pm.addNestedPass<func::FuncOp>(createRewriteToBatchReduceGemmPass());

      // Convert all higher level dialects to TPP.
      pm.addNestedPass<func::FuncOp>(createConvertLinalgToTppPass());

      pm.addPass(createConvertVNNIToTppPass());

      // Lower all TPP ops.
      if (tppToLoops)
        pm.addNestedPass<func::FuncOp>(createConvertTppToLoopsPass());
      else
        pm.addNestedPass<func::FuncOp>(createConvertTppToXsmmPass());
    }

    // Lower all Check ops.
    pm.addPass(createConvertCheckToLoopsPass());

    // Postprocess generated loops.
    // Perform LICM before function calls are generated to ensure that ops
    // which map directly to functions also get moved outside of loops, if
    // possible. This approach assumes that the function calls do not have any
    // side effects and can be safely moved outside of loop body.
    pm.addPass(createLoopInvariantCodeMotionPass());
    pm.addPass(createRaiseToParallelLoopPass());
    pm.addPass(createParallelLoopFusionPass());

    // Lower all XSMM ops.
    pm.addPass(createConvertXsmmToFuncPass());

    // General postprocessing.
    pm.addPass(bufferization::createBufferHoistingPass());
    pm.addPass(bufferization::createBufferDeallocationPass());
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::tpp::createDefaultTppPass(bool tppLoops, bool linalgLoops) {
  return std::make_unique<DefaultTppPasses>(tppLoops, linalgLoops);
}
