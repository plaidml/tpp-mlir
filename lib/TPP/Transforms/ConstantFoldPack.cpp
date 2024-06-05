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
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_CONSTANTFOLDPACK
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

struct ConstantFoldPack
    : public tpp::impl::ConstantFoldPackBase<ConstantFoldPack> {

  void runOnOperation() override {
    auto module = getOperation();
    auto *ctx = &getContext();

    // Apply pack canonicalization to fold trivial cases.
    RewritePatternSet packFolderPatterns(&getContext());
    tensor::PackOp::getCanonicalizationPatterns(packFolderPatterns, ctx);
    (void)applyPatternsAndFoldGreedily(module, std::move(packFolderPatterns));

    // Collect operations that pack constants.
    SmallVector<tensor::PackOp> packsToFold;
    module->walk([&](tensor::PackOp packOp) {
      auto constOp = packOp.getSource().getDefiningOp<arith::ConstantOp>();
      if (!constOp)
        return WalkResult::skip();
      // Must be a dense constant.
      auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue());
      if (!denseAttr)
        return WalkResult::skip();

      // Bail out if the pack is used as a writing operation i.e.,
      // the destination is not a tensor.empty.
      if (!packOp.getDest().getDefiningOp<tensor::EmptyOp>())
        return WalkResult::skip();
      // Pack destination must have static shape.
      if (!packOp.getDestType().hasStaticShape())
        return WalkResult::skip();

      // Pack with padding is not supported currently.
      // TODO: Add tensor.pad folder pattern when available.
      if (packOp.getPaddingValue())
        return WalkResult::skip();

      packsToFold.push_back(packOp);

      return WalkResult::advance();
    });

    // Go through intermediate lowering through Linalg operations.
    IRRewriter rewriter(ctx);
    for (auto pack : packsToFold) {
      FailureOr<linalg::LowerPackResult> res =
          linalg::lowerPack(rewriter, pack);
      assert(succeeded(res) && "failed intermediate pack lowering");
    }

    // Apply Linalg constant folders to cleanup lowered packs.
    // TODO: Add tensor.pad folder pattern when available.
    RewritePatternSet constantFolderPatterns(&getContext());
    linalg::populateConstantFoldLinalgOperations(
        constantFolderPatterns, [](OpOperand *) -> bool { return true; });
    (void)applyPatternsAndFoldGreedily(module,
                                       std::move(constantFolderPatterns));
  }
};

} // namespace
