//===- TppCompilerPipeline.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/TppPasses.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "Standalone/TppPasses.h.inc"

namespace {

struct TppCompilerPipeline
    : public TppCompilerPipelineBase<TppCompilerPipeline> {
  void runOnOperation() override;
};

void TppCompilerPipeline::runOnOperation() {
  OpPassManager pm("builtin.module");
  if (failed(runPipeline(pm, getOperation())))
    signalPassFailure();
  return;
}

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::tpp::createTppCompilerPipeline() {
  return std::make_unique<TppCompilerPipeline>();
}
