//===- MainClosure.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/Stdx/StdxOps.h"
#include "Standalone/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "Standalone/Passes.h.inc"

namespace {

struct MainClosure : public MainClosureBase<MainClosure> {
  MainClosure() = default;
  void runOnOperation() override {

    func::FuncOp main = getOperation();
    if (main.getName() != "main")
      return;

    SmallVector<Value> closureArgs;
    Value closureRes = nullptr;
    for (BlockArgument argument : main.getArguments()) {
      if (main.getArgAttr(argument.getArgNumber(), "stdx.const"))
        continue;
      if (main.getArgAttr(argument.getArgNumber(), "stdx.res"))
        closureRes = argument;
      else 
        closureArgs.push_back(argument);
    }

    // no input, return.
    if (closureArgs.size() == 0)
      return;

    // no output, return.
    if (!closureRes)
      return;

    // Build the closure.
    Block *mainBlock = &main.getRegion().front();
    ImplicitLocOpBuilder builder =
        ImplicitLocOpBuilder::atBlockBegin(main.getLoc(), &main.front());
    stdx::ClosureOp closure =
        builder.create<stdx::ClosureOp>(closureRes, closureArgs);
 
    // Map closure region arguments with the main arguments.
    BlockAndValueMapping mapper;
    mapper.map(closureArgs, closure.getRegion().getArguments());

    // Get all the operations in main and clone them into the closure.
    // We skip closure itself and the func::return.
    ImplicitLocOpBuilder closureBuilder = ImplicitLocOpBuilder::atBlockBegin(
        closure.getLoc(), &closure.getRegion().front());
    SmallVector<Value> closureResults;
    for (Operation &op : mainBlock->getOperations()) {
      if (isa<func::ReturnOp>(op)) {
        for (auto opValue : op.getOperands())
          closureResults.push_back(mapper.lookupOrDefault(opValue));
        continue;
      }
      if (isa<stdx::ClosureOp>(op))
        continue;
      closureBuilder.clone(op, mapper);
    }

    // Fix up yield and return.
    stdx::YieldOp yield =
        cast<stdx::YieldOp>(closure.getRegion().front().getTerminator());
    yield->setOperands(closureResults);
    func::ReturnOp returnOp =
        cast<func::ReturnOp>(main.getRegion().front().getTerminator());
    returnOp->setOperands(closure->getResults());

    return;
  } // end runOnOperation
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createMainClosurePass() {
  return std::make_unique<MainClosure>();
}
