//===ConvertPackOptimization.cpp --------------------------------*----- C++-*-===//
////
//// Part of the LLVM Project, under the Apache License v2.0 with LLVM
/// Exceptions. / See https://llvm.org/LICENSE.txt for license information. /
/// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
////
////===----------------------------------------------------------------------===//
//
//
#include "TPP/BuilderUtils.h"
#include "TPP/Dialect/Tpp/TppOps.h"
#include "TPP/Dialect/Tpp/TppUtils.h"
#include "TPP/Passes.h"
#include "TPP/TransformUtils.h"
#include "TPP/Transforms.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {

typedef struct MatchResult {
  bool haveMatch;
  int index;
} MatchResult;

SmallVector<MatchResult> innerDimMatchesIndex(tensor::PackOp packOp) {
  SmallVector<MatchResult> result;
  int innerTiles = 0;
  for (size_t i = 0; i < packOp.getSource().getType().getShape().size(); i++) {
    if (packOp.getInnerDimsPos().size() > 0) {
      bool haveMatch = false;
      for (size_t j = 0; j < packOp.getInnerDimsPos().size(); j++) {
        if (i == (size_t)packOp.getInnerDimsPos()[j]) {
          MatchResult tempResult;
          tempResult.haveMatch = true;
          haveMatch = true;
          tempResult.index = innerTiles++;
          result.push_back(tempResult);
          break;
        }
      }
      if (!haveMatch) {
        MatchResult tempResult;
        tempResult.haveMatch = false;
        tempResult.index = -1;
        result.push_back(tempResult);
      }
    }
  }
  return result;
}

struct ConvertPackOptimizationOp : public OpRewritePattern<tensor::PackOp> {
  using OpRewritePattern<tensor::PackOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::PackOp packOp,
                                PatternRewriter &rewriter) const override {
    if (packOp.getStaticInnerTiles().size() <= 0) {
      return failure();
    }
    auto shape = packOp.getSource().getType().getShape();
    int numLoops = shape.size();

    SmallVector<MatchResult> haveMatched = innerDimMatchesIndex(packOp);
    for (int i = 0; i < numLoops; i++) {
      if (haveMatched[i].haveMatch) {
        if (shape[i] < packOp.getStaticInnerTiles()[haveMatched[i].index]) {
          return failure();
        }
      }
    }

    auto zero = getConstIndex(rewriter, 0);
    auto one = getConstIndex(rewriter, 1);

    SmallVector<Value> lbs;
    SmallVector<Value> ubs;
    SmallVector<Value> steps;

    for (int i = 0; i < numLoops; i++) {
      lbs.push_back(zero);
      steps.push_back(one);
    }

    for (int i = 0; i < numLoops; i++) {

      if (haveMatched[i].haveMatch) {
        ubs.push_back(getConstIndex(
            rewriter,
            shape[i] / packOp.getStaticInnerTiles()[haveMatched[i].index]));

      } else {
        ubs.push_back(getConstIndex(rewriter, shape[i]));
      }
    }
    Value pos;

    SmallVector<Value> reduc = {
        packOp.getDest(),
    };

    auto bodyBuilder = [&](OpBuilder &builder, Location, Value iv,
                           MutableArrayRef<Value> reduc) {};

    auto loopNest = mlir::scf::buildLoopNest(
        rewriter, packOp.getLoc(), lbs, ubs, steps, reduc,
        [&reduc, &pos, bodyBuilder, &packOp,
         &numLoops](OpBuilder &rewriter, Location loc, ValueRange localIvs,
                    ValueRange iterArgs) -> scf::ValueVector {
          reduc.assign(iterArgs.begin(), iterArgs.end());

          SmallVector<OpFoldResult> offsets;
          SmallVector<MatchResult> haveMatched = innerDimMatchesIndex(packOp);
          for (int i = 0; i < numLoops; i++) {

            if (haveMatched[i].haveMatch) {
              Value muliOp = rewriter.create<arith::MulIOp>(
                  loc, localIvs[i],
                  getConstIndex(
                      rewriter,
                      packOp.getStaticInnerTiles()[haveMatched[i].index]));
              offsets.push_back(muliOp);
            } else {
              offsets.push_back(localIvs[i]);
            }
          }
          SmallVector<OpFoldResult> strides;
          for (int i = 0; i < numLoops; i++) {
            strides.push_back(rewriter.getIndexAttr(1));
          }

          SmallVector<OpFoldResult> sizes;

          for (int i = 0; i < numLoops; i++) {
            if (haveMatched[i].haveMatch) {
              sizes.push_back(rewriter.getIndexAttr(
                  packOp.getStaticInnerTiles()[haveMatched[i].index]));
            } else {
              sizes.push_back(rewriter.getIndexAttr(1));
            }
          }

          auto tensorExtractType =
              tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
                  packOp.getStaticInnerTiles().size(),
                  packOp.getSource().getType().cast<RankedTensorType>(),
                  offsets, sizes, strides);

          auto tensorExtract = rewriter.create<tensor::ExtractSliceOp>(
              loc, tensorExtractType.cast<RankedTensorType>(),
              packOp.getSource(), offsets, sizes, strides);

          SmallVector<OpFoldResult> insertSliceOffsets;
          for (int i = 0; i < numLoops; i++) {
            int indirection = i;
            if (packOp.getOuterDimsPerm().size() > 0) {
              indirection = packOp.getOuterDimsPerm()[i];
            }
            insertSliceOffsets.push_back(localIvs[indirection]);
          }
          for (size_t i = numLoops; i < packOp.getDestRank(); i++) {
            insertSliceOffsets.push_back(rewriter.getIndexAttr(0));
          }
          SmallVector<OpFoldResult> insertSliceSizes;
          for (int i = 0; i < numLoops; i++) {
            insertSliceSizes.push_back(rewriter.getIndexAttr(1));
          }
          for (size_t i = numLoops; i < packOp.getDestRank(); i++) {
            insertSliceSizes.push_back(rewriter.getIndexAttr(
                packOp.getStaticInnerTiles()[i - numLoops]));
          }

          SmallVector<OpFoldResult> insertSliceStrides(
              packOp.getDestRank(), rewriter.getIndexAttr(1));
          auto insertSliceOp = rewriter.create<tensor::InsertSliceOp>(
              loc, tensorExtract.getResult(), iterArgs[0], insertSliceOffsets,
              insertSliceSizes, insertSliceStrides);
          return {insertSliceOp};
        });
    rewriter.replaceAllUsesWith(packOp.getResult(),
                                loopNest.loops[0].getResults()[0]);
    return success();
  }
};

void populatePackOptimizationPatterns(RewritePatternSet &patterns) {
  // clang-format off
     patterns.add<ConvertPackOptimizationOp>(patterns.getContext());
  // clang-format on
}

struct ConvertPackOptimization : public ConvertPackOptimizationBase<ConvertPackOptimization> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populatePackOptimizationPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::tpp::createConvertPackOptimization() {
  return std::make_unique<ConvertPackOptimization>();
}
