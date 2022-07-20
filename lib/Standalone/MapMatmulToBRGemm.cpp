//===- MapMatmulToBRGemm.cpp -------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/Xsmm/XsmmDialect.h"
#include "Standalone/Dialect/Xsmm/XsmmOps.h"
#include "Standalone/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "Standalone/Passes.h.inc"

namespace {

struct MapMatmulToBRGemm : public MapMatmulToBRGemmBase<MapMatmulToBRGemm> {
  void runOnOperation() override {
    // MLIRContext *ctx = getOperation().getContext();
    // RewritePatternSet patterns(ctx);
    // patterns.add<DoIt>(ctx);
    //(void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // end namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createMapMatmulToBRGemmPass() {
  return std::make_unique<MapMatmulToBRGemm>();
}
