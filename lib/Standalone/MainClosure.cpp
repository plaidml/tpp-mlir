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
    SmallVector<Type> closureArgsTypes;
    SmallVector<unsigned> argsIndex;
    for (BlockArgument argument : main.getArguments()) {
      if (main.getArgAttr(argument.getArgNumber(), "std.const"))
        continue;
      closureArgs.push_back(argument);
      argsIndex.push_back(argument.getArgNumber());
      closureArgsTypes.push_back(argument.getType());
    }

    ImplicitLocOpBuilder builder =
        ImplicitLocOpBuilder::atBlockBegin(main.getLoc(), &main.front());
    ClosureOp closure = builder.create<stdx::ClosureOp>(closureArgs);
    ImplicitLocOpBuilder bodyBuilder = ImplicitLocOpBuilder::atBlockBegin(
        closure.getLoc(), &closure.getRegion().front());

    bodyBuilder.create<arith::ConstantIndexOp>(32);
    bodyBuilder.create<stdx::YieldOp>(closure.getRegionIterArgs());

    return;

  } // end runOnOperation
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::tpp::createMainClosurePass() {
  return std::make_unique<MainClosure>();
}
