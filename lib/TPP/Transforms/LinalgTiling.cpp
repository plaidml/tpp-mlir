//===- LinalgTiling.cpp -----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements parallel loop insertion for tiling.
//
//===----------------------------------------------------------------------===//
#include "TPP/IR/TilingUtils.h"
#include "TPP/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <iostream>
#include <list>
#include <set>
#define DEBUG_TYPE "linalg-tiling"

namespace mlir {
namespace tpp {
#define GEN_PASS_DECL_LINALGTILING
#define GEN_PASS_DEF_LINALGTILING
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

using namespace mlir;
using namespace mlir::tpp;

namespace mlir {
namespace tpp {
struct LinalgOpTiling : OpRewritePattern<linalg::BatchReduceMatmulOp> {
  using OpRewritePattern<linalg::BatchReduceMatmulOp>::OpRewritePattern;

  LinalgOpTiling(MLIRContext *ctx, LinalgTilingOptions tilingoptions)
      : OpRewritePattern(ctx), options(tilingoptions) {}

  LogicalResult matchAndRewrite(linalg::BatchReduceMatmulOp linalgOp,
                                PatternRewriter &rewriter) const override {

    if (!linalgOp.hasPureBufferSemantics())
      return failure();
    //  Get the MXN tile shape from the user input
    std::vector<int64_t> tileShapeM(options.mTileShape.begin(),
                                    options.mTileShape.end());
    std::vector<int64_t> tileShapeN(options.nTileShape.begin(),
                                    options.nTileShape.end());
    std::vector<int64_t> finaltile(3);
    std::vector<int64_t> resulttile(2);

    if (tileShapeM.size() != 2 || tileShapeN.size() != 2)
	   return failure(); 

    if (tileShapeM[1] != tileShapeN[0])
	    return failure();

    finaltile[0] = tileShapeM[0];
    finaltile[1] = tileShapeN[1];
    finaltile[2] = tileShapeM[1];
    resulttile[0] = tileShapeM[0];
    resulttile[1] = tileShapeN[1];

    SmallVector<int64_t> boundariesOne{1,
                                       static_cast<long>(tileShapeM.size() - 1),
                                       static_cast<long>(finaltile.size() - 1)};

    SmallVector<int64_t> tileSizesIndex{static_cast<long>(tileShapeM.size()),
                                        static_cast<long>(tileShapeN.size()),
                                        static_cast<long>(resulttile.size())};
    std::vector<int64_t> tempTileM(tileShapeM.begin(), tileShapeM.end());
    std::vector<int64_t> tempTileN(tileShapeN.begin(), tileShapeN.end());
    SmallVector<std::vector<int64_t>> tileshapes{tempTileM, tempTileN,
                                                 resulttile};
    std::vector<int> swap_i = {0, 2, 1};
    size_t i = 0;
    std::map<int, std::map<int, Value>> inductionVars;
    scf::ForOp innermostForLoop;
    scf::ForOp reductionForLoop;

    // Creating the tiled loops
    for (auto itrShapeM = finaltile.begin(); itrShapeM != finaltile.end();
         itrShapeM++, i++) {
      int index = swap_i[i] / boundariesOne[swap_i[i]];
      int offset = swap_i[i] / (finaltile.size() - 1);

      int operandSize =
          dyn_cast<MemRefType>(linalgOp.getOperand(index).getType())
              .getShape()
              .size();
      int effectiveOffset = operandSize - tileSizesIndex[index] + offset;
      auto upperBound =
          dyn_cast<MemRefType>(linalgOp.getOperand(index).getType())
              .getShape()[effectiveOffset];
      Location loc = linalgOp.getLoc();
      Value zeroCst = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value ubCst = rewriter.create<arith::ConstantIndexOp>(loc, upperBound);
      Value stepCst = rewriter.create<arith::ConstantIndexOp>(loc, upperBound/(*itrShapeM));
      scf::ForOp loopOp = rewriter.create<scf::ForOp>(linalgOp.getLoc(),
                                                      zeroCst, ubCst, stepCst);
      rewriter.setInsertionPointToStart(loopOp.getBody());
      int indexTwo = offset;
      int operandSizeTwo =
          dyn_cast<MemRefType>(linalgOp.getOperand(indexTwo).getType())
              .getShape()
              .size();
      int effectiveOffsetTwo = operandSizeTwo - tileSizesIndex[index] + index;

      inductionVars[index][effectiveOffset] = loopOp.getInductionVar();

      inductionVars[indexTwo][effectiveOffsetTwo] = loopOp.getInductionVar();
      int indexThree = resulttile.size();
      int effectiveOffsetThree =
          index +
          dyn_cast<MemRefType>(linalgOp.getOperand(indexThree).getType())
              .getShape()
              .size() -
          tileSizesIndex[indexThree];
      if (inductionVars[indexThree][effectiveOffsetThree] == NULL) {
        inductionVars[indexThree][effectiveOffsetThree] =
            loopOp.getInductionVar();
      }
      innermostForLoop = loopOp;
      if ((finaltile.size() - 1) == (i + 1)) {
        Value zeroCst1 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        Value ubCst1 = rewriter.create<arith::ConstantIndexOp>(
            loc, dyn_cast<MemRefType>(linalgOp.getOperand(0).getType())
                     .getShape()[0]);
        Value stepCst1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
        scf::ForOp redloopOp = rewriter.create<scf::ForOp>(
            linalgOp.getLoc(), zeroCst1, ubCst1, stepCst1);
        rewriter.setInsertionPointToStart(redloopOp.getBody());
        reductionForLoop = redloopOp;
      }
    }

    // Creating subviews
    SmallVector<std::vector<int64_t>> tiles = {tileShapeM, tileShapeN};
    for (size_t i = 0; i < linalgOp.getNumOperands(); i++) {
      SmallVector<int64_t> indices;

      auto input = linalgOp.getOperand(i);
      auto operandType = input.getType();
      SmallVector<OpFoldResult> offsets;
      size_t k = 0;
      auto tileItr = tileshapes[i].begin();
      auto tensorShape = dyn_cast<MemRefType>(operandType).getShape();
      SmallVector<OpFoldResult> shape;
      SmallVector<OpFoldResult> strides;
      for (size_t j = 0; j < tensorShape.size(); j++) {
        if (j < tensorShape.size() - tileSizesIndex[i]) {
          if (j == ((tensorShape.size() - tileSizesIndex[i]) - 1) &&
              i < (linalgOp.getNumOperands() - 1)) {
            offsets.push_back(reductionForLoop.getInductionVar());
            indices.push_back(tensorShape[j] / 32);
            shape.push_back(rewriter.getIndexAttr(tensorShape[j] / 32));
            strides.push_back(rewriter.getIndexAttr(1));

          } else {
            offsets.push_back(rewriter.getIndexAttr(0));
            indices.push_back(tensorShape[j]);
            shape.push_back(rewriter.getIndexAttr(tensorShape[j]));
            strides.push_back(rewriter.getIndexAttr(1));
          }
        } else {
          shape.push_back(rewriter.getIndexAttr(tensorShape[j] / (*tileItr)));
          indices.push_back(tensorShape[j] / (*tileItr));
          strides.push_back(rewriter.getIndexAttr(1));
          offsets.push_back(
              inductionVars[i][tensorShape.size() - tileSizesIndex[i] + k]);
          k++;
          tileItr++;
        }
      }

      auto subview = rewriter.create<memref::SubViewOp>(
          linalgOp.getLoc(), nullptr /* dyn_cast<MemRefType>(subviewType)*/,
          input, offsets, shape, strides);
      linalgOp.setOperand(i, subview);
    }

    rewriter.setInsertionPoint(innermostForLoop.getBody(),
                               std::prev(innermostForLoop.getBody()->end(), 1));
    auto clone = rewriter.clone(*linalgOp);
    linalgOp.replaceAllUsesWith(clone);
    if (linalgOp->use_empty())
      rewriter.eraseOp(linalgOp);
    return success();
  }

private:
  LinalgTilingOptions options;
};

void populateLinalgTilingPatterns(RewritePatternSet &patterns,
                                  LinalgTilingOptions options) {
  patterns.add<LinalgOpTiling>(patterns.getContext(), options);
}

struct LinalgTiling : public tpp::impl::LinalgTilingBase<LinalgTiling> {

  using LinalgTilingBase::LinalgTilingBase;

  void runOnOperation() override {
    LinalgTilingOptions options{mTileShape, nTileShape};
    RewritePatternSet patterns(&getContext());
    populateLinalgTilingPatterns(patterns, options);
    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;

    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                       config);
  }
};
} // namespace tpp
} // namespace mlir
