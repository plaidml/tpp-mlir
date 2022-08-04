//===- UndoMainClosure.cpp --------------------------------------*- C++ -*-===//
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

using namespace mlir;

#define GEN_PASS_CLASSES
#include "Standalone/Passes.h.inc"

namespace {

struct UndoMainClosure : UndoMainClosureBase<UndoMainClosure> {
  UndoMainClosure() = default;
  void runOnOperation() override {
    func::FuncOp main = getOperation();
    if (main.getName() != "main")
      return;
    stdx::ClosureOp closureOp = nullptr;
    main.walk([&](stdx::ClosureOp currentOp) { closureOp = currentOp; });
    if (!closureOp)
      return;

    Block *closureBlock = &closureOp.getRegion().front();
    OpBuilder builder(closureOp);

    BlockAndValueMapping mapper;
    mapper.map(closureOp.getRegion().getArguments(),
               closureOp.getIterOperands());

    SmallVector<Value> mainResults;
    for (Operation &op : closureBlock->getOperations()) {
      if (isa<stdx::YieldOp>(op)) {
        for (auto opValue : op.getOperands())
          mainResults.push_back(mapper.lookupOrDefault(opValue));
        continue;
      }
      builder.clone(op, mapper);
    }

    func::ReturnOp returnOp =
        cast<func::ReturnOp>(main.getRegion().front().getTerminator());
    returnOp->setOperands(mainResults);

    return;
  } // end runOnOperation
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createUndoMainClosurePass() {
  return std::make_unique<UndoMainClosure>();
}
