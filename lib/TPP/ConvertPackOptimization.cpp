//===ConvertPackOptimization.cpp -------------------------------*----C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. / See https://llvm.org/LICENSE.txt for license information. /
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

#include "TPP/BuilderUtils.h"
#include "TPP/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"
#include <iostream>
namespace {

struct ConvertPackOptimizationOp : public OpRewritePattern<tensor::PackOp> {
  using OpRewritePattern<tensor::PackOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::PackOp packOp,
                                PatternRewriter &rewriter) const override {
    if (packOp.getStaticInnerTiles().size() <= 0)
      return failure();
    auto shape = packOp.getSourceType().getShape();
    int numLoops = shape.size();

    for (size_t i = 0; i < packOp.getInnerDimsPos().size() - 1; i++) {
      if (packOp.getInnerDimsPos()[i] >= packOp.getInnerDimsPos()[i + 1]) {
        return failure();
      }
    }

    auto zero = getConstIndex(rewriter, 0);
    auto one = getConstIndex(rewriter, 1);

    SmallVector<Value> lbs(numLoops, zero);
    SmallVector<Value> ubs;
    SmallVector<Value> steps(numLoops, one);

    std::map<int, int> tiledDims;
    for (auto tiledDim : llvm::enumerate(packOp.getInnerDimsPos())) {
      tiledDims[tiledDim.value()] = tiledDim.index();
    }

    for (int i = 0; i < numLoops; i++) {
      if (tiledDims.count(i) && shape[i] != ShapedType::kDynamic &&
          packOp.getStaticInnerTiles()[tiledDims[i]] != ShapedType::kDynamic) {
        ubs.push_back(getConstIndex(
            rewriter, shape[i] / packOp.getStaticInnerTiles()[tiledDims[i]]));

      } else {
        ubs.push_back(getConstIndex(rewriter, shape[i]));
      }
    }

    auto loopNest = mlir::scf::buildLoopNest(
        rewriter, packOp.getLoc(), lbs, ubs, steps, packOp.getDest(),
        [&packOp, &numLoops,
         &tiledDims](OpBuilder &rewriter, Location loc, ValueRange localIvs,
                     ValueRange iterArgs) -> scf::ValueVector {
          SmallVector<OpFoldResult> offsets;

          for (int i = 0; i < numLoops; i++) {
            if (tiledDims.count(i)) {
              Value muliOp = rewriter.create<arith::MulIOp>(
                  loc, localIvs[i],
                  getConstIndex(rewriter,
                                packOp.getStaticInnerTiles()[tiledDims[i]]));
              offsets.push_back(muliOp);
            } else {
              offsets.push_back(localIvs[i]);
            }
          }
          SmallVector<OpFoldResult> strides(numLoops, rewriter.getIndexAttr(1));

          SmallVector<OpFoldResult> sizes;

          for (int i = 0; i < numLoops; i++) {
            if (tiledDims.count(i)) {
              sizes.push_back(rewriter.getIndexAttr(
                  packOp.getStaticInnerTiles()[tiledDims[i]]));
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
          SmallVector<OpFoldResult> insertSliceSizes(numLoops,
                                                     rewriter.getIndexAttr(1));

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
    rewriter.replaceOp(packOp, loopNest.loops[0].getResults()[0]);
    return success();
  }
};

void populatePackOptimizationPatterns(RewritePatternSet &patterns) {
  // clang-format off
     patterns.add<ConvertPackOptimizationOp>(patterns.getContext());
  // clang-format on
}

struct ConvertPackOptimization
    : public ConvertPackOptimizationBase<ConvertPackOptimization> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populatePackOptimizationPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createConvertPackOptimization() {
  return std::make_unique<ConvertPackOptimization>();
}
