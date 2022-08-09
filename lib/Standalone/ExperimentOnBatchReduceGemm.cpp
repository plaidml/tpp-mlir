//===-ExperimentOnBatchReduceGemm.cpp  ---------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "Standalone/Passes.h.inc"

namespace {

struct BrgemmOutput : public OpRewritePattern<linalg::ReduceBatchMatmulOp> {
  using OpRewritePattern<linalg::ReduceBatchMatmulOp>::OpRewritePattern;

  FailureOr<Value> locateRelayoutPoint(Value value) const { return failure(); }

  LogicalResult matchAndRewrite(linalg::ReduceBatchMatmulOp brgemmOp,
                                PatternRewriter &rewriter) const override {
    if (!brgemmOp.hasTensorSemantics())
      return failure();
    // locate a potential relayout and check if the input of the relayout
    // is a broadcast, if so optimize.
    FailureOr<Value> relayout =
        locateRelayoutPoint(brgemmOp.getOutputOperands()[0]->get());
    return failure();
  }
};

void populateExperiments(RewritePatternSet &patterns) {
  patterns.add<BrgemmOutput>(patterns.getContext());
}

struct ExperimentOnBatchReduceGemm
    : public ExperimentOnBatchReduceGemmBase<ExperimentOnBatchReduceGemm> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateExperiments(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // end namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createExperimentOnBatchReduceGemmPass() {
  return std::make_unique<ExperimentOnBatchReduceGemm>();
}
