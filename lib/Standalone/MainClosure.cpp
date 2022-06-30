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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace mlir;
using namespace mlir::tpp;
using namespace mlir::stdx;

#define GEN_PASS_CLASSES
#include "Standalone/Passes.h.inc"

namespace {

// TODO: I don't want to erase main args and return.

struct MainClosure : public MainClosureBase<MainClosure> {
  MainClosure() = default;
  void runOnOperation() override {
    ModuleOp module = getOperation();
    func::FuncOp main = module.lookupSymbol<func::FuncOp>("main");
    if (!main)
      assert(0);

    SmallVector<BlockArgument> args;
    llvm::BitVector argIndices(main.getNumArguments());
    SmallVector<Type> argTypes;
    for (BlockArgument arg : main.getArguments()) {
      if (!main.getArgAttr(arg.getArgNumber(), "stdx.const")) {
        args.push_back(arg);
        argIndices.set(arg.getArgNumber());
        argTypes.push_back(arg.getType());
      }
    }

    Block *origBlock = &main.front();
    Operation *firstOp = &origBlock->front();
    func::ReturnOp returnOp = cast<func::ReturnOp>(origBlock->getTerminator());

    auto builder = ImplicitLocOpBuilder::atBlockBegin(main.getLoc(), origBlock);
    FunctionType funcType =
        builder.getFunctionType(argTypes, main.getFunctionType().getResults());
    auto closure = builder.create<stdx::ClosureOp>(funcType);

    Region &bodyRegion = closure.body();
    Block *body = new Block();
    bodyRegion.push_back(body);

    auto &oldBodyOps = origBlock->getOperations();
    auto &newBodyOps = body->getOperations();
    newBodyOps.splice(std::prev(newBodyOps.end()), oldBodyOps,
                      Block::iterator(firstOp), std::prev(oldBodyOps.end()));

    builder.setInsertionPointToEnd(body);
    builder.create<stdx::YieldOp>(returnOp.operands());

    returnOp->setOperands({});

    for (BlockArgument arg : args) {
      BlockArgument newArg = body->addArgument(arg.getType(), arg.getLoc());
      arg.replaceAllUsesWith(newArg);
    }

    main.eraseArguments(argIndices);
    for (unsigned i = 0, e = main.getNumResults(); i < e; ++i)
      main.eraseResult(0);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::tpp::createMainClosurePass() {
  return std::make_unique<MainClosure>();
}
