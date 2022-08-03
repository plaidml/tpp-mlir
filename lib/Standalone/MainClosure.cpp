//===- MainClosure.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/Stdx/StdxOps.h"
#include "Standalone/Passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include <set>

using namespace mlir;
using namespace mlir::stdx;

#define GEN_PASS_CLASSES
#include "Standalone/Passes.h.inc"

namespace {

struct MainClosure : public MainClosureBase<MainClosure> {
  MainClosure() = default;
  void runOnOperation() override {

    ModuleOp module = getOperation();
    func::FuncOp main = module.lookupSymbol<func::FuncOp>("main");
    if (!main)
      return;

    SmallVector<Value> closureArgs;
    SmallVector<unsigned> argsIndex;
    Value closureRes;
    for (BlockArgument argument : main.getArguments()) {
      if (main.getArgAttr(argument.getArgNumber(), "stdx.const"))
        continue;
      if (main.getArgAttr(argument.getArgNumber(), "stdx.res"))
        closureRes = argument;
      else {
        closureArgs.push_back(argument);
        argsIndex.push_back(argument.getArgNumber());
      }
    }

    BlockAndValueMapping mapper;
    Block *mainBlock = &main.getRegion().front();
    ImplicitLocOpBuilder builder =
        ImplicitLocOpBuilder::atBlockBegin(main.getLoc(), &main.front());
    ClosureOp closure =
        builder.create<stdx::ClosureOp>(closureRes, closureArgs);

    ImplicitLocOpBuilder closureBuilder = ImplicitLocOpBuilder::atBlockBegin(
        closure.getLoc(), &closure.getRegion().front());
    mapper.map(main.getArguments(), closure.getRegion().getArguments());

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

std::unique_ptr<OperationPass<ModuleOp>> mlir::tpp::createMainClosurePass() {
  return std::make_unique<MainClosure>();
}
