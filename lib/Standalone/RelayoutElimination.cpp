//===- RelayoutElimination.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/LinalgX/LinalgXDialect.h"
#include "Standalone/Dialect/LinalgX/LinalgXOps.h"
#include "Standalone/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "Standalone/Passes.h.inc"

namespace {

struct RelayoutElimination
    : public RelayoutEliminationBase<RelayoutElimination> {
  void runOnOperation() override;
};

void RelayoutElimination::runOnOperation() {

  SmallVector<Operation *> candidates;
  getOperation().walk([&](linalgx::Relayout relayoutOp) {
    if (relayoutOp.hasBufferSemantics())
      candidates.push_back(relayoutOp);
  });

  if (!candidates.size())
    return;

  // See how 'ParallelLoopFusion.cpp' uses walk
  // to check side effect in between operations.
  DominanceInfo domInfo;
  for (Operation *op : candidates) {
    linalgx::Relayout currentRoot = cast<linalgx::Relayout>(op);
    Value rootOperand = currentRoot.getInput();
    getOperation().walk([&](linalgx::Relayout candidateRelayout) {
      if (domInfo.properlyDominates(currentRoot.getOperation(),
                                    candidateRelayout.getOperation())) {
        Value candidateOperand = candidateRelayout.getInput();
        if (rootOperand == candidateOperand)
          llvm::errs() << "candidate: " << candidateRelayout << "\n";
      }
      return WalkResult::advance();
    });
  }
}

} // end namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createRelayoutEliminationPass() {
  return std::make_unique<RelayoutElimination>();
}
