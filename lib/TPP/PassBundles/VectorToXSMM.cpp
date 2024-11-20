//===- VectorToXSMM.cpp ------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Debug.h"

#include "TPP/PassBundles.h"
#include "TPP/PassUtils.h"

using namespace mlir;
using namespace mlir::tpp;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_VECTORTOXSMM
#include "TPP/PassBundles.h.inc"
} // namespace tpp
} // namespace mlir

#define DEBUG_TYPE "convert-vector-to-xsmm"

// Apply collection of vector-level passes that map vector patterns to
// libxsmm call pairs (dispatch, invoke).
struct VectorToXSMM : public tpp::impl::VectorToXSMMBase<VectorToXSMM>,
                    PassBundle<ModuleOp> {
  void runOnOperation() override {
    auto module = getOperation();

    // Initialize the pipeline if needed.
    // Otherwise, just run the cached one.
    if (pm.empty())
      constructPipeline();

    if (failed(runPipeline(pm, module))) {
      return signalPassFailure();
    }
  }

private:
  void constructPipeline() override {
    LLVM_DEBUG(llvm::dbgs() << "Adding vector-to-xsmm passes\n");
    // Not Implemented Yet.
  }
};
