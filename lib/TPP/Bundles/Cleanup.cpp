//===- Cleanup.cpp -----------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Bundles.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "TPP/PassUtils.h"

using namespace mlir;
using namespace mlir::tpp;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_CLEANUP
#include "TPP/Bundles.h.inc"
} // namespace tpp
} // namespace mlir

// A general cleanup pass that performs general IR normalization and
// generic optimizations without any lowering or any logical changes.
// Commonly applied after other major passes.
struct Cleanup : public tpp::impl::CleanupBase<Cleanup>, PassBundle<> {
  using CleanupBase::CleanupBase;

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
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }
};
