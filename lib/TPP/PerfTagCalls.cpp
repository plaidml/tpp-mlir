//===- PerfTagCalls.cpp ------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Perf/PerfOps.h"
#include "TPP/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

struct PerfTagCalls : public PerfTagCallsBase<PerfTagCalls> {
  void tagCallOps(perf::BenchOp benchOp) {
    for (auto &op : benchOp.getRegion().getOps()) {
      if (isa<func::CallOp>(op))
        perf::BenchOp::tagOp(&op);
    }
  }

  void runOnOperation() override {
    auto module = getOperation();
    IRRewriter rewriter(&getContext());
    module->walk([&](perf::BenchOp benchOp) { tagCallOps(benchOp); });
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createPerfTagCallsPass() {
  return std::make_unique<PerfTagCalls>();
}
