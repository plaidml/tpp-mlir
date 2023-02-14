//===- DecomposeLinalgOps.cpp - Test Linalg decomposition  ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Tpp/TppOps.h"
#include "TPP/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {
struct DecomposeLinalgOps
    : public DecomposeLinalgOpsPassBase<DecomposeLinalgOps> {
  void runOnOperation() override {
    RewritePatternSet decompositionPatterns(&getContext());
    linalg::populateDecomposeLinalgOpsPattern(decompositionPatterns);
    (void)applyPatternsAndFoldGreedily(getOperation(),
                                       std::move(decompositionPatterns));
    return;
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createDecomposeLinalgOpsPass() {
  return std::make_unique<DecomposeLinalgOps>();
}
