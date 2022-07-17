//===- TileConsumerAndFuseProducers.cpp --------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/Tpp/TppUtils.h"
#include "Standalone/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "Standalone/Passes.h.inc"

namespace {

struct FuseGenericOp : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  // Locate an element-wise operation and fuse if the producer
  // is a matmul.
  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {

    // Well.. avoid recursion by checking the parent operation...
    Operation *parentOp = linalgOp->getParentOp();
    if (parentOp && isa<scf::ForOp>(parentOp))
      return failure();

    // hook only element-wise operation with tensor semantics.
    if (!linalgOp.hasTensorSemantics() || !linalg::isElementwise(linalgOp))
      return failure();

    // further restrict to single operand operations produced by a matmul.
    linalg::OpOperandVector operands = linalgOp.getInputOperands();
    if (operands.size() != 1)
      return failure();
    linalg::LinalgOp producer =
        dyn_cast_or_null<linalg::LinalgOp>(operands[0]->get().getDefiningOp());
    if (!producer || !tpp::isMarkedWithTpp(producer, "tpp.matmul"))
      return failure();

    // tile and fuse.
    scf::SCFTilingOptions options;
    // TODO: better tiling strategy.
    options.setTileSizes({32, 32});
    scf::TileConsumerAndFuseProducersUsingSCFForOp tileAndFuse(
        linalgOp.getContext(), options);
    TilingInterface tilingInterfaceOp =
        cast<TilingInterface>(linalgOp.getOperation());
    FailureOr<scf::SCFTileAndFuseResult> tileAndFuseResult =
        tileAndFuse.returningMatchAndRewrite(tilingInterfaceOp, rewriter);
    if (failed(tileAndFuseResult))
      return failure();
    return success();
  }
};

void populateFusionPatterns(RewritePatternSet &patterns) {
  patterns.add<FuseGenericOp>(patterns.getContext());
}

struct TileConsumerAndFuseProducers
    : TileConsumerAndFuseProducersBase<TileConsumerAndFuseProducers> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
    linalg::registerTilingInterfaceExternalModels(registry);
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateFusionPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // end namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createTileConsumerAndFuseProducersPass() {
  return std::make_unique<TileConsumerAndFuseProducers>();
}
