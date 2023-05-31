//===- DeallocReturns.cpp ----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

// Convert buffers from heap to stack allocation.
struct DeallocFuncReturn : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    Location loc = callOp.getLoc();
    auto results = callOp.getResults();
    if (results.size() == 0)
      return rewriter.notifyMatchFailure(callOp, "Expected call return values");

    SmallVector<Value> buffs;
    for (auto result : results) {
      if (result.getType().isa<MemRefType>())
        buffs.push_back(result);
    }
    if (buffs.size() == 0)
      return rewriter.notifyMatchFailure(callOp, "Expected memref returns");

    auto terminator = callOp->getParentRegion()->begin()->getTerminator();
    for (auto &buf : buffs) {
      // Do not deallocate if the buffer is returned from the current region.
      if (llvm::any_of(terminator->getOperands(),
                       [&](Value op) { return op == buf; }))
        continue;

      // Do nothing if there is a deallocator already.
      if (llvm::any_of(buf.getUsers(), [](Operation *user) {
            return isa<memref::DeallocOp>(user);
          }))
        continue;

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(terminator);
      rewriter.create<memref::DeallocOp>(loc, buf);
    }

    return success();
  }
};

struct DeallocReturns : public DeallocReturnsBase<DeallocReturns> {
  DeallocReturns() = default;

  void runOnOperation() override {
    RewritePatternSet patterns(getOperation().getContext());
    patterns.add<DeallocFuncReturn>(patterns.getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createDeallocReturnsPass() {
  return std::make_unique<DeallocReturns>();
}
