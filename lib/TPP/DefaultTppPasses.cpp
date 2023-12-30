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
#include "TPP/Dialect/Transform/LinalgXTransformOps.h"
#include "TPP/Dialect/Xsmm/XsmmDialect.h"
#include "TPP/PassUtils.h"

using namespace mlir;
using namespace mlir::tpp;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_DEFAULTTPPPASSES
#include "TPP/Passes.h.inc"
#define GEN_PASS_DEF_LINALGLOWERING
#include "TPP/Passes.h.inc"
#define GEN_PASS_DEF_CLEANUP
#include "TPP/Passes.h.inc"
#define GEN_PASS_DEF_LOCALDIALECTSLOWERING
#include "TPP/Passes.h.inc"
#define GEN_PASS_DEF_POSTPROCESSING
#include "TPP/Passes.h.inc"
#define GEN_PASS_DEF_TPPMAPPING
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

// A general cleanup pass that performs general IR normalization and
// generic optimizations without any lowering or any logical changes.
// Commonly applied after other major passes.
struct Cleanup : public tpp::impl::CleanupBase<Cleanup>,
                 UtilityPassBase<func::FuncOp> {
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

    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }
};

// Lower all local dialects (XSMM, check etc.) to standard dialects
// and function calls.
struct LocalDialectsLowering
    : public tpp::impl::LocalDialectsLoweringBase<LocalDialectsLowering>,
      UtilityPassBase<ModuleOp> {

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

    pm.addNestedPass<func::FuncOp>(createConvertCheckToLoops());
    pm.addNestedPass<func::FuncOp>(createConvertPerfToLoops());

    // Note that LICM should be performed before any function calls are
    // generated
    // to ensure that ops which map directly to functions also get moved outside
    // of loops, if possible. This approach assumes that the function calls do
    // not have any side effects and can be safely moved outside of loop body.
    pm.addNestedPass<func::FuncOp>(createLoopInvariantCodeMotionPass());
    // Run cleanup after LICM to allow CSE to eliminate common operations now
    // that they are hoisted out of loops.
    pm.addNestedPass<func::FuncOp>(createCleanup());

    pm.addPass(createConvertXsmmToFunc());
    pm.addPass(createConvertPerfToFunc());
  }
};

// Apply various postprocessing passes such as LICM, parallel loop fusion,
// buffer deallocation, general cleanup etc.
struct Postprocessing : public tpp::impl::PostprocessingBase<Postprocessing>,
                        UtilityPassBase<func::FuncOp> {
  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry
        .insert<bufferization::BufferizationDialect,
                memref::MemRefDialect,
                scf::SCFDialect>();
    // clang-format on
    check::registerBufferizableOpInterfaceExternalModels(registry);
    perf::registerBufferizableOpInterfaceExternalModels(registry);
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

    // Postprocess buffers.
    pm.addPass(bufferization::createBufferHoistingPass());

    // Run general cleanup to normalize IR.
    pm.addPass(createCleanup());
  }
};

// Apply collection of high-level passes that map operations to
// TPP-compatible forms.
struct TppMapping : public tpp::impl::TppMappingBase<TppMapping>,
                    UtilityPassBase<ModuleOp> {
  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry
        .insert<linalg::LinalgDialect,
                memref::MemRefDialect,
                scf::SCFDialect,
                tensor::TensorDialect>();
    // clang-format on
    check::registerBufferizableOpInterfaceExternalModels(registry);
    perf::registerBufferizableOpInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    auto module = getOperation();

    // Initialize the pipeline if needed.
    // Otherwise, just run the cached one.
    if (pm.empty())
      constructPipeline();

    if (failed(runPipeline(pm, module))) {
      llvm::dbgs() << "Failed tpp mapping\n";
      return signalPassFailure();
    }
  }

private:
  void constructPipeline() override {
    pm.clear();

    // Preprocess convolutions.
    pm.addPass(createConvInitSimplify());
    pm.addPass(createCleanup());

    // Convert ops to packed layouts.
    pm.addPass(createPackConv2DNhwcHwcf());
    pm.addPass(createPackConv2DNchwFchw());
    pm.addPass(createRewriteConvToMatmulOrBrgemm());
    pm.addPass(createPackMatmul());
    pm.addPass(createPackVNNI());

    // Postprocess packing.
    // Run only canonicalizer at this stage as full cleanup (mostly CSE) can
    // mess up tensor producer-consumer chains used for analysis in the
    // following passes.
    pm.addPass(createPropagatePackUnPack());
    pm.addPass(createConstantFoldPack());
    pm.addPass(createSimplifyAndCanonicalizePack());

    pm.addPass(createCleanup());
    pm.addPass(createTileConsumerAndFuseProducers());
    pm.addPass(createSimplifyAndCanonicalizePack());
    pm.addPass(createCleanup());
  }
};

// Lower Linalg to into combination of standard and local dialects.
struct LinalgLowering : public tpp::impl::LinalgLoweringBase<LinalgLowering>,
                        UtilityPassBase<func::FuncOp> {

  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry
        .insert<xsmm::XsmmDialect,
                scf::SCFDialect,
                memref::MemRefDialect>();
    // clang-format on
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

    pm.addPass(createConvertLinalgToXsmm());
    pm.addPass(createCombineXsmmOpPass());
    pm.addPass(createFoldXsmmFlags());
    pm.addPass(createVerifyXsmmCalls());
  }
};

// The default pipeline for TPP.
struct DefaultTppPasses
    : public tpp::impl::DefaultTppPassesBase<DefaultTppPasses>,
      UtilityPassBase<ModuleOp> {
  using DefaultTppPassesBase::DefaultTppPassesBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    // Add all custom TPP dialects.
    registry.insert<xsmm::XsmmDialect>();
    registry.insert<check::CheckDialect>();
    registry.insert<perf::PerfDialect>();
    linalgx::registerTransformDialectExtension(registry);
    check::registerBufferizableOpInterfaceExternalModels(registry);
    perf::registerBufferizableOpInterfaceExternalModels(registry);

    // Add all core MLIR dialects as the default TPP passes may contain any
    // combination of other passes.
    registerAllDialects(registry);
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

    // Default pipeline does not support transforms yet
    pm.addPass(createTransformDropSchedule());

    if (linalgToLoops) {
      // Lower linalg directly to loops.
      // Skip all TPP transformations.
      // Generalize tensor.pack and tensor.unpack.
      pm.addPass(createLowerPacksAndUnPacks());
      pm.addNestedPass<func::FuncOp>(createDecomposeAggregatedOps());
      pm.addPass(createBufferize());
      pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
      pm.addNestedPass<func::FuncOp>(createCleanup());
    } else {
      // Convert linalg.batch_matmul to linalg.matmul.
      pm.addPass(createRewriteBatchMatmulToMatmul());

      // Applies a set of passes at the linalg level to fuse and pack.
      pm.addPass(createTppMapping());

      // Generalize tensor.pack and tensor.unpack.
      pm.addPass(createLowerPacksAndUnPacks());
      pm.addNestedPass<func::FuncOp>(createCleanup());

      // Decompose Aggregated operations. These ops currently do not
      // bufferize. Once this is possible we can move this pass after
      // bufferization.
      pm.addNestedPass<func::FuncOp>(createDecomposeAggregatedOps());

      // Bufferize: tensor->memref.
      pm.addPass(createBufferize());

      // Lower all Tile operations.
      pm.addNestedPass<func::FuncOp>(createLinalgLowering());
      pm.addNestedPass<func::FuncOp>(createCleanup());
    }

    // Convert forAll to parallel loops should run after bufferization
    // as scf.parallel does not handle tensor.
    pm.addPass(createConvertForAllToParallelOp());

    // Covert all local TPP-related dialects.
    pm.addPass(createLocalDialectsLowering());

    // Clean up after the default pipeline.
    pm.addNestedPass<func::FuncOp>(createPostprocessing());
  }
};

} // namespace
