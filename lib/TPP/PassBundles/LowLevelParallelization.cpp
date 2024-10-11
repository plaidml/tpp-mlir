//===- LowLevelParallelization.cpp -------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/PassBundles.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "TPP/PassUtils.h"

using namespace mlir;
using namespace mlir::tpp;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_LOWLEVELPARALLELIZATION
#include "TPP/PassBundles.h.inc"
} // namespace tpp
} // namespace mlir

// Low level parallelization, 2D blocking, AMX config
struct LowLevelParallelization
    : public tpp::impl::LowLevelParallelizationBase<LowLevelParallelization>,
      PassBundle<ModuleOp> {
  using LowLevelParallelizationBase::LowLevelParallelizationBase;

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
    // Note that LICM should be performed before any function calls are
    // generated
    // to ensure that ops which map directly to functions also get moved outside
    // of loops, if possible. This approach assumes that the function calls do
    // not have any side effects and can be safely moved outside of loop body.
    pm.addNestedPass<func::FuncOp>(createLoopInvariantCodeMotionPass());
    // Run cleanup after LICM to allow CSE to eliminate common operations now
    // that they are hoisted out of loops.
    pm.addPass(createCleanup());

    mlir::tpp::SCFParallelLoopTilingOptions tilingOptions;
    tilingOptions.tileSizes = parallelTaskGrid;
    pm.addPass(createSCFParallelLoopTiling(tilingOptions));
    pm.addPass(createLoopInvariantCodeMotionPass());
  }
};
