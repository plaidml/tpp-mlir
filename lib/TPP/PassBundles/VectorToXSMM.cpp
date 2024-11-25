//===- VectorToXSMM.cpp ------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Conversion/ConvertVectorToXsmm/ConvertTranspose.h"
#include "TPP/Conversion/ConvertVectorToXsmm/ConvertVectorToXsmm.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/PDLExtension/PDLExtensionOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
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
    pm.addPass(createInsertTranspose());
    pm.addPass(createConvertVectorToXsmm());
  }
};
