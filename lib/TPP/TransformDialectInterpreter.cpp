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
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

struct TransformDialectInterpreter
    : TransformDialectInterpreterBase<TransformDialectInterpreter> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    for (auto op :
         module.getBody()->getOps<transform::TransformOpInterface>()) {
      if (failed(transform::applyTransforms(
              module, op,
              transform::TransformOptions().enableExpensiveChecks(true))))
        return signalPassFailure();
    }
  }
};

struct TransformDropSchedule
    : TransformDropScheduleBase<TransformDropSchedule> {
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

struct DefaultSchedule : DefaultScheduleBase<DefaultSchedule> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto builder =
        ImplicitLocOpBuilder::atBlockEnd(module->getLoc(), module.getBody());
    OperationState opState(module->getLoc(), "transform.sequence");
    opState.addRegion();
    opState.addAttribute(
        "failure_propagation_mode",
        transform::FailurePropagationModeAttr::get(
            builder.getContext(), transform::FailurePropagationMode::Suppress));
    Region *region = opState.regions.back().get();
    Type pdlType = pdl::OperationType::get(builder.getContext());
    opState.addTypes({pdlType});
    Operation *created = builder.create(opState);
    transform::SequenceOp sequence = cast<transform::SequenceOp>(created);
    // WHY?
    region = &sequence.getBody();
    Block *bodyBlock = new Block();
    bodyBlock->addArguments(TypeRange{pdlType}, {module->getLoc()});
    region->push_back(bodyBlock);
    bodyBlock->dump();

    builder.setInsertionPointToStart(bodyBlock);
    builder.create<transform::YieldOp>(module->getLoc(),
                                       region->getArguments());
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::tpp::createTransformDialectInterpreterPass() {
  return std::make_unique<TransformDialectInterpreter>();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::tpp::createTransformDropSchedulePass() {
  return std::make_unique<TransformDropSchedule>();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::tpp::createDefaultSchedulePass() {
  return std::make_unique<DefaultSchedule>();
}
