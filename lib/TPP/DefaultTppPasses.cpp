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
#include "TPP/Dialect/Tpp/BufferizableOpInterfaceImpl.h"
#include "TPP/Dialect/Tpp/TppDialect.h"
#include "TPP/Dialect/Transform/LinalgXTransformOps.h"
#include "TPP/Dialect/Xsmm/XsmmDialect.h"
#include "TPP/PassUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

// A general cleanup pass that performs general IR normalization and
// generic optimizations without any lowering or any logical changes.
// Commonly applied after other major passes.
struct CleanupPass : public CleanupBase<CleanupPass>,
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
struct LocalDialectsLoweringPass
    : public LocalDialectsLoweringBase<LocalDialectsLoweringPass>,
      UtilityPassBase<ModuleOp> {
  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry
        .insert<affine::AffineDialect,
                arith::ArithDialect,
                func::FuncDialect,
                memref::MemRefDialect,
                check::CheckDialect,
                perf::PerfDialect,
                scf::SCFDialect,
                tensor::TensorDialect,
                xsmm::XsmmDialect,
                LLVM::LLVMDialect>();
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

    pm.addNestedPass<func::FuncOp>(createConvertCheckToLoopsPass());
    pm.addNestedPass<func::FuncOp>(createConvertPerfToLoopsPass());

    // Note that LICM should be performed before any function calls are
    // generated
    // to ensure that ops which map directly to functions also get moved outside
    // of loops, if possible. This approach assumes that the function calls do
    // not have any side effects and can be safely moved outside of loop body.
    pm.addNestedPass<func::FuncOp>(createLoopInvariantCodeMotionPass());
    // Run cleanup after LICM to allow CSE to eliminate common operations now
    // that they are hoisted out of loops.
    pm.addNestedPass<func::FuncOp>(createCleanupPass());

    pm.addPass(createConvertXsmmToFuncPass());
    pm.addPass(createConvertPerfToFuncPass());
  }
};

// Apply various postprocessing passes such as LICM, parallel loop fusion,
// buffer deallocation, general cleanup etc.
struct PostprocessingPass : public PostprocessingBase<PostprocessingPass>,
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
    pm.addPass(createCleanupPass());
  }
};

// Apply collection of high-level passes that map operations to
// TPP-compatible forms.
struct TppMappingPass : public TppMappingBase<TppMappingPass>,
                        UtilityPassBase<ModuleOp> {
  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry
        .insert<linalg::LinalgDialect,
                memref::MemRefDialect,
                scf::SCFDialect,
                tensor::TensorDialect,
                tpp::TppDialect>();
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
    pm.addPass(createConvInitSimplifyPass());
    pm.addPass(createCleanupPass());
    pm.addPass(createPackConv2DNhwcHwcfPass({32, 32}));
    pm.addPass(createPackConv2DNchwFchwPass({32, 32}));
    pm.addPass(createRewriteConvToMatmulOrBrgemmPass());

    // Convert ops to packed layouts.
    pm.addPass(createPackMatmulPass({32, 32, 32}));
    pm.addPass(createPackVNNIPass());

    // Postprocess packing.
    // Run only canonicalizer at this stage as full cleanup (mostly CSE) can
    // mess up tensor producer-consumer chains used for analysis in the
    // following passes.
    pm.addPass(createPropagatePackUnPackPass());
    pm.addPass(createConstantFoldPackPass());
    pm.addPass(createSimplifyAndCanonicalizePackPass());

    pm.addPass(createCleanupPass());
    pm.addPass(createTileConsumerAndFuseProducersPass());
    pm.addPass(createSimplifyAndCanonicalizePackPass());
    pm.addPass(createCleanupPass());
  }
};

// Convert all matching operations to TPP.
struct TppConversionPass : public TppConversionBase<TppConversionPass>,
                           UtilityPassBase<func::FuncOp> {
  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry
        .insert<linalg::LinalgDialect,
                scf::SCFDialect,
                tpp::TppDialect>();
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

    // Convert all higher level dialects to TPP.
    pm.addPass(createConvertLinalgToTpp());
    pm.addPass(createCombineTppPass());
  }
};

// Lower TPP to into combination of standard and local dialects.
struct TppLoweringPass : public TppLoweringBase<TppLoweringPass>,
                         UtilityPassBase<func::FuncOp> {
  TppLoweringPass() : TppLoweringPass(false, true){};
  TppLoweringPass(bool tppToLoops, bool linalgToXsmm) {
    this->tppToLoops = tppToLoops;
    this->linalgToXsmm = linalgToXsmm;
  };

  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry
        .insert<xsmm::XsmmDialect,
                scf::SCFDialect,
                memref::MemRefDialect,
                tpp::TppDialect>();
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

    // Lower all TPP ops.
    if (tppToLoops)
      pm.addPass(createConvertTppToLoopsPass());
    else {
      // Memref to tpp conversion patterns.
      pm.addPass(createConvertMemRefToXsmm());
      if (linalgToXsmm) {
        // Linalg to Xsmm conversion patterns.
        pm.addPass(createConvertLinalgToXsmmPass());
      } else {
        // Tpp to Xsmm conversion patterns.
        pm.addPass(createConvertTppToXsmmPass());
      }
    }
  }
};

// The default pipeline for TPP.
struct DefaultTppPasses : public DefaultTppPassesBase<DefaultTppPasses>,
                          UtilityPassBase<ModuleOp> {
  DefaultTppPasses() : DefaultTppPasses(false, false, true){};
  DefaultTppPasses(bool tppToLoops, bool linalgToLoops, bool linalgToXsmm) {
    this->tppToLoops = tppToLoops;
    this->linalgToLoops = linalgToLoops;
    this->linalgToXsmm = linalgToXsmm;
  };

  void getDependentDialects(DialectRegistry &registry) const override {
    // Add all custom TPP dialects.
    registry.insert<tpp::TppDialect>();
    registry.insert<xsmm::XsmmDialect>();
    registry.insert<check::CheckDialect>();
    registry.insert<perf::PerfDialect>();
    linalgx::registerTransformDialectExtension(registry);
    check::registerBufferizableOpInterfaceExternalModels(registry);
    perf::registerBufferizableOpInterfaceExternalModels(registry);
    tpp::registerBufferizableOpInterfaceExternalModels(registry);

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
    pm.addPass(createTransformDropSchedulePass());

    if (linalgToLoops) {
      // Lower linalg directly to loops.
      // Skip all TPP transformations.
      // Generalize tensor.pack and tensor.unpack.
      pm.addPass(createLowerPacksAndUnPacks());
      pm.addNestedPass<func::FuncOp>(createDecomposeAggregatedOpsPass());
      pm.addPass(createBufferizePass());
      pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
      pm.addNestedPass<func::FuncOp>(createCleanupPass());
    } else {
      // Convert linalg.batch_matmul to linalg.matmul.
      pm.addPass(createRewriteBatchMatmulToMatmulPass());

      // Applies a set of passes at the linalg level to fuse and pack.
      pm.addPass(createTppMappingPass());

      // Generalize tensor.pack and tensor.unpack.
      pm.addPass(createLowerPacksAndUnPacks());
      pm.addNestedPass<func::FuncOp>(createCleanupPass());

      // Decompose Aggregated operations. These ops currently do not
      // bufferize. Once this is possible we can move this pass after
      // bufferization.
      pm.addNestedPass<func::FuncOp>(createDecomposeAggregatedOpsPass());

      // Lower linalg operations to TPP.
      if (!linalgToXsmm) {
        pm.addNestedPass<func::FuncOp>(createTppConversionPass());
        pm.addNestedPass<func::FuncOp>(createCleanupPass());
      }

      // Bufferize: tensor->memref.
      pm.addPass(createBufferizePass());

      // Lower all Tile operations.
      pm.addNestedPass<func::FuncOp>(
          createTppLoweringPass(tppToLoops, linalgToXsmm));
      pm.addNestedPass<func::FuncOp>(createCleanupPass());
    }

    // Convert forAll to parallel loops should run after bufferization
    // as scf.parallel does not handle tensor.
    pm.addPass(createConvertForAllToParallelOpPass());

    // Covert all local TPP-related dialects.
    pm.addPass(createLocalDialectsLoweringPass());

    // Clean up after the default pipeline.
    pm.addNestedPass<func::FuncOp>(createPostprocessingPass());
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::tpp::createCleanupPass() {
  return std::make_unique<CleanupPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::tpp::createLocalDialectsLoweringPass() {
  return std::make_unique<LocalDialectsLoweringPass>();
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createPostprocessingPass() {
  return std::make_unique<PostprocessingPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::tpp::createTppMappingPass() {
  return std::make_unique<TppMappingPass>();
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createTppConversionPass() {
  return std::make_unique<TppConversionPass>();
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createTppLoweringPass(bool linalgToLoops, bool linalgToXsmm) {
  return std::make_unique<TppLoweringPass>(linalgToLoops, linalgToXsmm);
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::tpp::createDefaultTppPass(bool tppLoops, bool linalgLoops,
                                bool linalgToXsmm) {
  return std::make_unique<DefaultTppPasses>(tppLoops, linalgLoops,
                                            linalgToXsmm);
}
