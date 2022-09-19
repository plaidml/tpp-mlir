//===- TransformDialectInterpreter.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a test pass that interprets Transform dialect operations in
// the module.
//
//===----------------------------------------------------------------------===//

#include "Standalone/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "Standalone/Passes.h.inc"

namespace {

struct TransformDialectInterpreter
    : TransformDialectInterpreterBase<TransformDialectInterpreter> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    transform::TransformState state(
        module.getBodyRegion(), module,
        transform::TransformOptions().enableExpensiveChecks(
            true /*enableExpensiveChecks*/));
    for (auto op : module.getBody()->getOps<transform::TransformOpInterface>())
      if (failed(state.applyTransform(op).checkAndReport()))
        return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::tpp::createTransformDialectInterpreterPass() {
  return std::make_unique<TransformDialectInterpreter>();
}
