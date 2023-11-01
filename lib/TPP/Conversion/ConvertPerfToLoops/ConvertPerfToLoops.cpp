//===- ConvertPerfToLoops.cpp ------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Perf/PerfOps.h"
#include "TPP/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace mlir;
using namespace mlir::perf;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_CONVERTPERFTOLOOPS
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

struct ConvertBenchToLoops : public OpRewritePattern<perf::BenchOp> {
  using OpRewritePattern<perf::BenchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(perf::BenchOp benchOp,
                                PatternRewriter &rewriter) const override {
    auto loc = benchOp.getLoc();
    auto benchYield = benchOp.getRegion().front().getTerminator();
    assert(dyn_cast_or_null<perf::YieldOp>(benchYield) &&
           "expect perf.yield in perf.bench");

    auto numIters = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), benchOp.getNumIters());

    // Create benchmark loop up to perf.bench numIters.
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto loop = rewriter.create<scf::ForOp>(loc, zero, numIters, one,
                                            benchOp.getIterArgs());
    if (benchOp.getIterArgs().empty()) {
      // Erase the default loop yield, it will be inserted later.
      rewriter.eraseOp(loop.getRegion().front().getTerminator());
    }

    // Move perf.bench region inside the loop.
    rewriter.mergeBlocks(&benchOp.getRegion().front(), loop.getBody());

    // Wrap the benchmark kernel in timer calls.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(loop.getBody());
    auto timer = rewriter.create<perf::StartTimerOp>(
        loc, TimerType::get(rewriter.getContext()));
    rewriter.setInsertionPointToEnd(loop.getBody());
    auto delta = rewriter.create<perf::StopTimerOp>(loc, rewriter.getF64Type(),
                                                    timer.getTimer());

    // Move all perf.sink ops after the timer to prevent influencing
    // measurements.
    for (auto &op : loop.getRegion().getOps()) {
      if (isa<perf::SinkOp>(op))
        op.moveAfter(delta.getOperation());
    }

    // Store measured time delta on each iteration.
    rewriter.create<memref::StoreOp>(loc, delta.getDelta(), benchOp.getDeltas(),
                                     loop.getInductionVar());

    // Replace uses of bench args within the benchmark body with their
    // equivalent loop-carried variables.
    assert((benchOp.getIterArgs().size() == loop.getRegionIterArgs().size()) &&
           "expect equal number of loop-carried variables");
    for (auto [benchArg, loopArg] :
         llvm::zip_equal(benchOp.getIterArgs(), loop.getRegionIterArgs()))
      replaceAllUsesInRegionWith(benchArg, loopArg, loop.getRegion());

    // Pass perf.yield values through the scf.yield.
    rewriter.setInsertionPointToEnd(loop.getBody());
    rewriter.create<scf::YieldOp>(loc, benchYield->getOperands());
    rewriter.eraseOp(benchYield);

    // Swap bench results with loop results.
    assert((benchOp.getBodyResults().size() == loop.getResults().size()) &&
           "expect equal number of loop-carried variables");
    for (auto [benchRes, loopRes] :
         llvm::zip_equal(benchOp.getBodyResults(), loop.getResults()))
      benchRes.replaceAllUsesWith(loopRes);

    rewriter.eraseOp(benchOp);
    return success();
  }
};

void populatePerfToLoopsPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertBenchToLoops>(patterns.getContext());
}

struct ConvertPerfToLoops
    : public tpp::impl::ConvertPerfToLoopsBase<ConvertPerfToLoops> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populatePerfToLoopsPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace
