//===- TileConsumerAndFuseProducers.cpp --------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Tpp/TppUtils.h"
#include "TPP/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

// Check if loopRange is statically known.
static bool isStaticRange(Range loopRange) {
  Optional<int64_t> offsetAsInt = getConstantIntValue(loopRange.offset);
  if (!offsetAsInt)
    return false;
  Optional<int64_t> sizeAsInt = getConstantIntValue(loopRange.size);
  if (!sizeAsInt)
    return false;
  Optional<int64_t> strideAsInt = getConstantIntValue(loopRange.stride);
  return static_cast<bool>(strideAsInt);
}

static int64_t getSizeRange(Range loopRange) {
  assert(isStaticRange(loopRange));
  return *getConstantIntValue(loopRange.size) -
         *getConstantIntValue(loopRange.offset);
}

static LogicalResult tileDivideIterationDomain(linalg::GenericOp linalgOp,
                                               ArrayRef<int64_t> tiles,
                                               OpBuilder &builder) {
  if (linalgOp.getNumLoops() != tiles.size())
    return failure();
  SmallVector<Range> iterationDomain =
      cast<TilingInterface>(linalgOp.getOperation())
          .getIterationDomain(builder);
  for (const auto &it : llvm::enumerate(iterationDomain)) {
    // fine, we are not tiling along this dimension.
    if (tiles[it.index()] == 0)
      continue;
    // require static loop range
    if (!isStaticRange(it.value()))
      return failure();
    int64_t sizeRange = getSizeRange(it.value());
    // fail if the tail size equals the range
    // or is not a full tile.
    if (tiles[it.index()] == sizeRange)
      return failure();
    if (sizeRange % tiles[it.index()] != 0)
      return failure();
  }
  return success();
}

struct FuseGenericOp : public OpRewritePattern<linalg::GenericOp> {
  FuseGenericOp(MLIRContext *context, ArrayRef<int64_t> tileSizes,
                PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::GenericOp>(context, benefit),
        tileSizes(tileSizes) {}

  bool isInplace(linalg::LinalgOp linalgOp) const {
    return linalgOp.getNumDpsInputs() == 0;
  }

  // Locate an element-wise operation and fuse if the producer
  // is a matmul.
  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {

    // hook only element-wise operation with tensor semantics.
    if (!linalgOp.hasTensorSemantics() || !linalg::isElementwise(linalgOp))
      return failure();

    // further restrict to single result operations.
    OpOperandVector operands = isInplace(linalgOp)
                                   ? linalgOp.getDpsInitOperands()
                                   : linalgOp.getDpsInputOperands();
    if (operands.size() != 1)
      return failure();
    linalg::LinalgOp producer =
        dyn_cast_or_null<linalg::LinalgOp>(operands[0]->get().getDefiningOp());
    if (!producer)
      return failure();

    if (producer.getNumParallelLoops() != tileSizes.size()) {
      return rewriter.notifyMatchFailure(
          producer,
          "expect tile sizes to be equal to number of parallel loops");
    }

    if (failed(tileDivideIterationDomain(linalgOp, tileSizes, rewriter)))
      return rewriter.notifyMatchFailure(producer, "wrong tile sizes");

    // tile and fuse.
    scf::SCFTileAndFuseOptions options;
    options.tilingOptions.setTileSizes(tileSizes);
    TilingInterface tilingInterfaceOp =
        cast<TilingInterface>(linalgOp.getOperation());
    FailureOr<scf::SCFTileAndFuseResult> tileAndFuseResult =
        tileConsumerAndFuseProducerGreedilyUsingSCFForOp(
            rewriter, tilingInterfaceOp, options);
    if (failed(tileAndFuseResult))
      return failure();
    rewriter.replaceOp(linalgOp, tileAndFuseResult->loops[0].getResults());
    return success();
  }
  ArrayRef<int64_t> tileSizes;
};

void populateFusionPatterns(RewritePatternSet &patterns,
                            ArrayRef<int64_t> tileSizes) {
  patterns.add<FuseGenericOp>(patterns.getContext(), tileSizes);
}

struct TileConsumerAndFuseProducers
    : TileConsumerAndFuseProducersBase<TileConsumerAndFuseProducers> {
  TileConsumerAndFuseProducers() = default;
  TileConsumerAndFuseProducers(ArrayRef<int64_t> tileSizes) {
    this->tileSizes = tileSizes;
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
    linalg::registerTilingInterfaceExternalModels(registry);
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateFusionPatterns(patterns, tileSizes);
    // fold unit-extent dims for linalg on tensors.
    linalg::populateFoldUnitExtentDimsPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // end namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createTileConsumerAndFuseProducersPass() {
  return std::make_unique<TileConsumerAndFuseProducers>();
}
