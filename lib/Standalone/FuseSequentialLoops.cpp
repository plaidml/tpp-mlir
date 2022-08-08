//===- FuseSequentialLoops.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "Standalone/Passes.h.inc"

namespace {

void naivelyFuseSequentialOps(Region &region) { return; }

struct FuseSequentialLoops
    : public FuseSequentialLoopsBase<FuseSequentialLoops> {
  void runOnOperation() override {
    getOperation()->walk([&](Operation *child) {
      for (Region &region : child->getRegions())
        naivelyFuseSequentialOps(region);
    });
  }
};

} // end namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createFuseSequentialLoopsPass() {
  return std::make_unique<FuseSequentialLoops>();
}
