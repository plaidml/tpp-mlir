//===- DefaultPipeline.cpp ---------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"

#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"

#include "TPP/Dialect/Check/BufferizableOpInterfaceImpl.h"
#include "TPP/Dialect/Check/CheckDialect.h"
#include "TPP/Dialect/Perf/BufferizableOpInterfaceImpl.h"
#include "TPP/Dialect/Perf/PerfDialect.h"
#include "TPP/Dialect/Perf/PerfOps.h"
#include "TPP/Dialect/Tpp/BufferizableOpInterfaceImpl.h"
#include "TPP/Dialect/Tpp/TppDialect.h"
#include "TPP/Dialect/Transform/LinalgXTransformOps.h"
#include "TPP/Dialect/Xsmm/XsmmDialect.h"
#include "TPP/PassUtils.h"
#include "TPP/Transforms/Utils/BuilderUtils.h"
#include "TPP/Transforms/Utils/TensorInit.h"
#include "TPP/Transforms/Utils/TensorInitFloat.h"
#include "TPP/Transforms/Utils/TensorInitInt.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <string>

using namespace mlir;
using namespace mlir::tpp;

// Print MLIR before lowering
llvm::cl::opt<std::string>
    printMLIR("print-mlir",
              llvm::cl::desc("Print MLIR to stdout (early, mid, late, llvm)"),
              llvm::cl::init(""));

// Lower Linalg directly to loops without TPP (for validation purposes)
llvm::cl::opt<bool> linalgToLoops("linalg-to-loops",
                                  llvm::cl::desc("Lower linalg to loops"),
                                  llvm::cl::init(false));

// Control parallelism.
llvm::cl::opt<bool>
    defParallel("def-parallel",
                llvm::cl::desc("Default pipeline - enable parallel execution"),
                llvm::cl::init(false));

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_DEFAULTPIPELINE
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

// Enum to control IR printing.
enum class PrintStage {
  None,
  Early, // After main generation, before optimization
  Mid,   // After initial TPP-related optimizations
  Late,  // After optimizaiton, before LLVM dialect
  LLVM,  // Final MLIR, in LLVM dialect
};

// Parses MLIR print stage
PrintStage parsePrintStage(StringRef stage) {
  return StringSwitch<PrintStage>(stage)
      .CaseLower("early", PrintStage::Early)
      .CaseLower("mid", PrintStage::Mid)
      .CaseLower("late", PrintStage::Late)
      .CaseLower("llvm", PrintStage::LLVM)
      .Default(PrintStage::None);
}

// The default lowering pipeline.
struct DefaultPipeline : public tpp::impl::DefaultPipelineBase<DefaultPipeline>,
                         UtilityPassBase<ModuleOp> {
  using DefaultPipelineBase::DefaultPipelineBase;

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

    // Add all core MLIR dialects as the default pipeline may contain any
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

    auto print = parsePrintStage(printMLIR);

    // Print IR of unoptimized kernel and main
    if (print == PrintStage::Early)
      pm.addPass(createPrintIRPass());

    if (!gpuBackend.empty()) {
      // Apply the custom GPU lowering pipeline
      pm.addPass(createGpuPipeline(GpuPipelineOptions{gpuBackend}));
    } else {
      // Apply the default preprocessing pass
      DefaultTppPassesOptions tppDefaultOptions{linalgToLoops};
      pm.addPass(createDefaultTppPasses(tppDefaultOptions));
    }

    if (print == PrintStage::Mid)
      pm.addPass(createPrintIRPass());

    // Partial Lowering
    pm.addPass(memref::createExpandStridedMetadataPass());
    pm.addNestedPass<func::FuncOp>(createConvertPerfToLoops());
    pm.addPass(tpp::createConvertPerfToFunc());
    pm.addPass(createConvertTensorToLinalgPass());
    pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
    if (defParallel)
      pm.addPass(createConvertSCFToOpenMPPass());
    pm.addPass(createConvertVectorToSCFPass());
    pm.addPass(arith::createArithExpandOpsPass());
    pm.addPass(createLowerAffinePass());

    // Print IR of optimized kernel and main
    if (print == PrintStage::Late)
      pm.addPass(createPrintIRPass());

    // Lower to LLVM
    pm.addPass(createConvertVectorToLLVMPass());
    pm.addPass(createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(createConvertSCFToCFPass());
    if (defParallel)
      pm.addPass(createConvertOpenMPToLLVMPass());
    pm.addPass(createConvertMathToLLVMPass());

    pm.addNestedPass<func::FuncOp>(createGpuAsyncRegionPass());
    pm.addPass(createGpuToLLVMConversionPass());
    GpuModuleToBinaryPassOptions gpuModuleToBinaryPassOptions;
    gpuModuleToBinaryPassOptions.compilationTarget = "fatbin";
    pm.addPass(createGpuModuleToBinaryPass(gpuModuleToBinaryPassOptions));
    pm.addPass(createAsyncToAsyncRuntimePass());
    pm.addPass(createAsyncRuntimeRefCountingPass());
    pm.addPass(createConvertAsyncToLLVMPass());

    pm.addPass(createConvertFuncToLLVMPass());

    pm.addNestedPass<func::FuncOp>(createArithToLLVMConversionPass());
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    pm.addNestedPass<func::FuncOp>(createCSEPass());
    pm.addPass(createReconcileUnrealizedCastsPass());

    pm.addPass(createConvertVulkanLaunchFuncToVulkanCallsPass());

    // Anything useful has been lowered by now.
    // Cleanup IR by removing any dead symbols.
    // This step aims to avoid errors caused by frontend leftovers.
    // See issue: #704
    pm.addPass(createSymbolDCEPass());

    // Print IR of kernel and main in LLVM dialect
    if (print == PrintStage::LLVM)
      pm.addPass(createPrintIRPass());
  }
};

} // namespace
