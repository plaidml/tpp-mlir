//===- ConvertXsmmToFunc.cpp -------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/Xsmm/XsmmOps.h"
#include "Standalone/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::xsmm;

#define GEN_PASS_CLASSES
#include "Standalone/Passes.h.inc"

namespace {

struct ConvertXsmmToFunc : public ConvertXsmmToFuncBase<ConvertXsmmToFunc> {
  void runOnOperation() override { return; }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::tpp::createConvertXsmmToFuncPass() {
  return std::make_unique<ConvertXsmmToFunc>();
}
