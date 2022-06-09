//===- LinalgConvertToTpp.cpp ------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/TppOps.h"
#include "Standalone/TppPasses.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "Standalone/TppPasses.h.inc"

#define DEBUG_TYPE "linalg-convert-to-tpp"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

namespace {

// TODO: evaluate transform dialect.

// Tiling function to remove all but the zero and first dimension.
static SmallVector<Value, 4> getTileSizes(OpBuilder &builder,
                                          linalg::LinalgOp linalgOp) {
  SmallVector<Value, 4> tppTiles;
  SmallVector<Range, 4> loopRanges =
      linalgOp.createLoopRanges(builder, linalgOp.getLoc());
  for (size_t i = 0; i < loopRanges.size(); i++)
    tppTiles.push_back(loopRanges[i].size);
  Value zeroVal = builder.create<arith::ConstantIndexOp>(linalgOp.getLoc(), 1);
  tppTiles[0] = zeroVal;
  tppTiles[1] = zeroVal;
  return tppTiles;
}

static MemRefType dropUnitDims(MemRefType inputType, ArrayRef<int64_t> offsets,
                               ArrayRef<int64_t> sizes,
                               ArrayRef<int64_t> strides) {
  Type rankReducedType = memref::SubViewOp::inferRankReducedResultType(
      0, inputType, offsets, sizes, strides);
  return canonicalizeStridedLayout(rankReducedType.cast<MemRefType>());
}

// Reduce rank for 'input' by dropping unit dimension.
static Value rankReducingSubviewDroppingUnitDims(OpBuilder &builder,
                                                 Location loc, Value input) {
  MemRefType inputType = input.getType().cast<MemRefType>();
  assert(inputType.hasStaticShape());
  SmallVector<int64_t> subViewOffsets(inputType.getRank(), 0);
  SmallVector<int64_t> subViewStrides(inputType.getRank(), 1);
  ArrayRef<int64_t> subViewSizes = inputType.getShape();
  MemRefType resultType =
      dropUnitDims(inputType, subViewOffsets, subViewSizes, subViewStrides);
  if (canonicalizeStridedLayout(resultType) ==
      canonicalizeStridedLayout(inputType))
    return input;
  return builder.create<memref::SubViewOp>(
      loc, resultType, input, subViewOffsets, subViewSizes, subViewStrides);
}

// Make the generic operation mappable to tpp by preserving
// the last and first dimension only.
LogicalResult reshape2D(linalg::GenericOp linalgOp) {
  if (!linalgOp.hasBufferSemantics())
    return linalgOp->emitError("Expect linalgOp with buffer semantics");

  // bail-out if we don't need to do tiling or all the dimensions
  // are not parallel.
  // TODO: restrict to only the tiling ones.
  if (linalgOp.getNumLoops() <= 2)
    return success();
  ArrayAttr iteratorTypes = linalgOp.iterator_types();
  if (!llvm::all_of(iteratorTypes,
                    [](Attribute type) { return isParallelIterator(type); }))
    return success();

  OpBuilder builder(linalgOp);
  OpBuilder::InsertionGuard guard(builder);
  linalg::LinalgTilingOptions linalgTilingOptions;
  linalgTilingOptions.setLoopType(linalg::LinalgTilingLoopType::ParallelLoops)
      .setTileSizeComputationFunction(getTileSizes);
  IRRewriter rewriter(builder);
  FailureOr<linalg::TiledLinalgOp> tiledOp =
      linalg::tileLinalgOp(rewriter, linalgOp, linalgTilingOptions);
  if (failed(tiledOp))
    return linalgOp->emitError("Failed to tile linalgOp");

  linalgOp->erase();
  return success();
}

struct ConvertGenericOpToTpp : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  bool isTppCallPrefix(std::string stringAttrAsStr) const {
    std::string delimiter = ".";
    std::string prefix =
        stringAttrAsStr.substr(0, stringAttrAsStr.find(delimiter));
    return prefix.compare("tpp") == 0;
  }

  LogicalResult rewriteToTppOp(linalg::GenericOp linalgOp,
                               ArrayRef<Value> operands,
                               PatternRewriter &rewriter) const {
    std::string libraryCall = linalgOp.getLibraryCallName();
    // TODO: find better way to express this.
    if (libraryCall.compare("tpp.identity") == 0) {
      rewriter.replaceOpWithNewOp<tpp::IdentityOp>(linalgOp, operands[0],
                                                   operands[1]);
      return success();
    }
    if (libraryCall.compare("tpp.relu") == 0) {
      rewriter.replaceOpWithNewOp<tpp::ReluOp>(linalgOp, operands[0],
                                               operands[1]);
      return success();
    }
    if (libraryCall.compare("tpp.add") == 0) {
      rewriter.replaceOpWithNewOp<tpp::AddOp>(linalgOp, operands[0],
                                              operands[1], operands[2]);
      return success();
    }
    if (libraryCall.compare("tpp.matmul") == 0) {
      rewriter.replaceOpWithNewOp<tpp::MatmulOp>(linalgOp, operands[0],
                                                 operands[1], operands[2]);
      return success();
    }
    return failure();
  }

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!linalgOp.hasBufferSemantics())
      return failure();

    if (!linalgOp.library_callAttr() || linalgOp.getNumLoops() != 2 ||
        !isTppCallPrefix(linalgOp.getLibraryCallName()))
      return failure();

    if (linalgOp->getNumResults() != 0)
      return failure();

    Location loc = linalgOp.getLoc();
    SmallVector<Value, 4> newOperands;
    for (Value operand : linalgOp->getOperands()) {
      ShapedType operandType = operand.getType().cast<ShapedType>();
      if (operandType.getRank() == 2) {
        newOperands.push_back(operand);
        continue;
      }
      if (operandType.getRank() > 2)
        newOperands.push_back(
            rankReducingSubviewDroppingUnitDims(rewriter, loc, operand));
      else
        assert(0 && "not implemented yet");
    }
    return rewriteToTppOp(linalgOp, newOperands, rewriter);
  }
};

void populateConvertLinalgToTppPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertGenericOpToTpp>(patterns.getContext());
}

// TODO: PatternRwriter does not work well with tiling. I suspect
// because the builder is not properly propagated. But investigate more.
struct ConvertLinalgToTpp : public ConvertLinalgToTppBase<ConvertLinalgToTpp> {
  void runOnOperation() override {
    getOperation().walk([&](linalg::GenericOp linalgOp) {
      // TODO: Check logical result.
      (void)reshape2D(linalgOp);
    });
    RewritePatternSet patterns(getOperation().getContext());
    populateConvertLinalgToTppPatterns(patterns);
    linalg::populateFoldUnitExtentDimsPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // end namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createConvertLinalgToTppPass() {
  return std::make_unique<ConvertLinalgToTpp>();
}
