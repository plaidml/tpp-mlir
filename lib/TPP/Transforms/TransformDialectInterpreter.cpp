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

#include "TPP/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace mlir::tpp;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_TRANSFORMDIALECTINTERPRETER
#include "TPP/Passes.h.inc"
#define GEN_PASS_DEF_TRANSFORMDROPSCHEDULE
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

struct TransformDialectInterpreter
    : tpp::impl::TransformDialectInterpreterBase<TransformDialectInterpreter> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    for (auto op :
         module.getBody()->getOps<transform::TransformOpInterface>()) {
      if (failed(transform::applyTransforms(
              module, op, {},
              transform::TransformOptions().enableExpensiveChecks(true))))
        return signalPassFailure();
    }
  }
};

struct TransformDropSchedule
    : tpp::impl::TransformDropScheduleBase<TransformDropSchedule> {
  void runOnOperation() override {
    getOperation()->walk<WalkOrder::PreOrder>([&](Operation *nestedOp) {
      if (isa<::mlir::transform::TransformOpInterface>(nestedOp)) {
        nestedOp->erase();
        return WalkResult::skip();
      }
      return WalkResult::advance();
    });
  }
};

} // namespace
