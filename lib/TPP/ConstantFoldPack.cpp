//===- ConstantFoldPack.cpp --------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

struct ConstantFoldPack : public ConstantFoldPackBase<ConstantFoldPack> {
  void runOnOperation() override {
    auto module = getOperation();
    transform::TrivialPatternRewriter rewriter(&getContext());
    module->walk([&](tensor::PackOp packOp) {
      Value sourcePack = packOp.getSource();
      auto cstOp = sourcePack.getDefiningOp<arith::ConstantOp>();
      if (!cstOp)
        return;
      auto cst = cstOp.getValue();
      if (!cst.isa<DenseElementsAttr>())
        return;
      auto oldDense = cast<DenseElementsAttr>(cst);
      if (!oldDense.isSplat())
        return;
      auto newDense = oldDense.reshape(packOp.getDestType());
      rewriter.setInsertionPoint(cstOp);
      auto newCstOp =
          rewriter.create<arith::ConstantOp>(cstOp.getLoc(), newDense);
      rewriter.replaceOp(packOp, newCstOp.getResult());
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::tpp::createConstantFoldPackPass() {
  return std::make_unique<ConstantFoldPack>();
}
