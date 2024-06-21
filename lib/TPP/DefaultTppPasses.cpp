//===- DefaultTppPasses.cpp --------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/PassBundles.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "TPP/Dialect/Check/BufferizableOpInterfaceImpl.h"
#include "TPP/Dialect/Check/CheckDialect.h"
#include "TPP/Dialect/Perf/BufferizableOpInterfaceImpl.h"
#include "TPP/Dialect/Perf/PerfDialect.h"
#include "TPP/Dialect/Xsmm/XsmmDialect.h"
#include "TPP/PassUtils.h"

using namespace mlir;
using namespace mlir::tpp;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_DEFAULTTPPPASSES
#include "TPP/PassBundles.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

// The default pipeline for TPP.
struct DefaultTppPasses
    : public tpp::impl::DefaultTppPassesBase<DefaultTppPasses>,
      PassBundle<ModuleOp> {
  using DefaultTppPassesBase::DefaultTppPassesBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    // Add all custom TPP dialects.
    registry.insert<xsmm::XsmmDialect>();
    registry.insert<check::CheckDialect>();
    registry.insert<perf::PerfDialect>();
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
    if (linalgToLoops) {
      // Lower linalg directly to loops.
      // Skip all TPP transformations.
      // Generalize tensor.pack and tensor.unpack.
      pm.addPass(createLowerPacksAndUnPacks());
      pm.addNestedPass<func::FuncOp>(createDecomposeAggregatedOps());
      pm.addPass(createBufferize());
      pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
      pm.addPass(createCleanup());
    } else {
      pm.addNestedPass<func::FuncOp>(createConvertAddInplacePass());
      // Convert linalg.batch_matmul to linalg.matmul.
      pm.addPass(createRewriteBatchMatmulToMatmul());

      // Applies a set of passes at the linalg level to fuse and pack.
      pm.addPass(createTppMapping());

      // Generalize tensor.pack and tensor.unpack.
      pm.addPass(createLowerPacksAndUnPacks());
      pm.addPass(createCleanup());

      // Decompose Aggregated operations. These ops currently do not
      // bufferize. Once this is possible we can move this pass after
      // bufferization.
      pm.addNestedPass<func::FuncOp>(createDecomposeAggregatedOps());

      // Bufferize: tensor->memref.
      pm.addPass(createBufferize());

      // Lower all Tile operations.
      pm.addNestedPass<func::FuncOp>(createLinalgLowering());
      pm.addPass(createCleanup());
    }
    // Low level parallelization passes.
    if (tileShapeM != 0 && tileShapeN != 0) {
      LowLevelParallelizationOptions LowLevelParallelization(
          LowLevelParallelizationOptions{tileShapeM, tileShapeN, shuffleOrder,
                                         outerParallelLoops});
      pm.addPass(createLowLevelParallelization(LowLevelParallelization));

      // Convert forAll to parallel loops should run after bufferization
      // as scf.parallel does not handle tensor.
      pm.addPass(createConvertForAllToParallelOp());
    } else {
      // FIXME remove as soon as the above code is fixed
      pm.addPass(createConvertForAllToParallelOp());
      mlir::tpp::SCFParallelLoopTilingOptions tilingOptions;
      tilingOptions.tileSizes = parallelTaskGrid;
      pm.addPass(createSCFParallelLoopTiling(tilingOptions));
    }
    pm.addPass(createCombineXsmmOpPass());
    pm.addNestedPass<func::FuncOp>(createLoopInvariantCodeMotionPass());
    pm.addPass(createFoldXsmmFlags());
    pm.addPass(createVerifyXsmmCalls());
    pm.addNestedPass<func::FuncOp>(createIntelAMXTileConfigInsertionPass());
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    pm.addNestedPass<func::FuncOp>(createLoopInvariantCodeMotionPass());
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    pm.addNestedPass<func::FuncOp>(createIntelAMXTileConfigHoistingPass());

    // Covert all local TPP-related dialects.
    pm.addPass(createLocalDialectsLowering());

    // Clean up after the default pipeline.
    pm.addNestedPass<func::FuncOp>(createPostprocessing());
  }
};

} // namespace
