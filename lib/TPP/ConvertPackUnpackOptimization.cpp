//===ConvertPackandUnpackOptimization.cpp ----------------------*----C++-*-===//
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

namespace {

template <typename Op>
struct ConvertPackUnpackOptimizationOp : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(Op packOp,
                                PatternRewriter &rewriter) const override {
    if (packOp.getStaticInnerTiles().size() <= 0)
      return failure();

    constexpr bool IsPack =
        std::is_same<mlir::tensor::PackOp, Op>::value == true;
    ArrayRef<int64_t> shape = IsPack ? packOp.getSourceType().getShape()
                                     : packOp.getDestType().getShape();
    int numLoops = shape.size();

    for (size_t i = 0; i < packOp.getInnerDimsPos().size() - 1; i++) {
      if (packOp.getInnerDimsPos()[i] >= packOp.getInnerDimsPos()[i + 1]) {
        return failure();
      }
    }

    for (size_t i = 0; i < packOp.getStaticInnerTiles().size(); i++) {
      if (shape[packOp.getInnerDimsPos()[i]] %
              packOp.getStaticInnerTiles()[i] !=
          0) {
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
        [&packOp, &numLoops, &tiledDims,
         &IsPack](OpBuilder &rewriter, Location loc, ValueRange localIvs,
                  ValueRange iterArgs) -> scf::ValueVector {
          SmallVector<OpFoldResult> extractSliceOffsets;
          SmallVector<OpFoldResult> extractSliceStrides;
          SmallVector<OpFoldResult> extractSliceSizes;
          SmallVector<OpFoldResult> insertSliceOffsets;
          SmallVector<OpFoldResult> insertSliceStrides;
          SmallVector<OpFoldResult> insertSliceSizes;

          if (IsPack) {
            for (int i = 0; i < numLoops; i++) {
              if (tiledDims.count(i)) {
                Value muliOp = rewriter.create<arith::MulIOp>(
                    loc, localIvs[i],
                    getConstIndex(rewriter,
                                  packOp.getStaticInnerTiles()[tiledDims[i]]));
                extractSliceOffsets.push_back(muliOp);
              } else {
                extractSliceOffsets.push_back(localIvs[i]);
              }
            }
            for (int i = 0; i < numLoops; i++)
              extractSliceStrides.push_back(rewriter.getIndexAttr(1));

            for (int i = 0; i < numLoops; i++) {
              if (tiledDims.count(i)) {
                extractSliceSizes.push_back(rewriter.getIndexAttr(
                    packOp.getStaticInnerTiles()[tiledDims[i]]));
              } else {
                extractSliceSizes.push_back(rewriter.getIndexAttr(1));
              }
            }
            for (size_t i = 0; i < packOp.getDestRank(); i++)
              insertSliceStrides.push_back(rewriter.getIndexAttr(1));

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
            for (int i = 0; i < numLoops; i++)
              insertSliceSizes.push_back(rewriter.getIndexAttr(1));

            for (size_t i = numLoops; i < packOp.getDestRank(); i++) {
              insertSliceSizes.push_back(rewriter.getIndexAttr(
                  packOp.getStaticInnerTiles()[i - numLoops]));
            }
          } else {

            for (int i = 0; i < numLoops; i++) {
              if (tiledDims.count(i)) {
                Value muliOp = rewriter.create<arith::MulIOp>(
                    loc, localIvs[i],
                    getConstIndex(rewriter,
                                  packOp.getStaticInnerTiles()[tiledDims[i]]));
                insertSliceOffsets.push_back(muliOp);
              } else {
                insertSliceOffsets.push_back(localIvs[i]);
              }
            }
            for (int i = 0; i < numLoops; i++)
              insertSliceStrides.push_back(rewriter.getIndexAttr(1));

            for (int i = 0; i < numLoops; i++) {
              if (tiledDims.count(i)) {
                insertSliceSizes.push_back(rewriter.getIndexAttr(
                    packOp.getStaticInnerTiles()[tiledDims[i]]));
              } else {
                insertSliceSizes.push_back(rewriter.getIndexAttr(1));
              }
            }
            for (int i = 0; i < numLoops; i++) {
              int indirection = i;
              if (packOp.getOuterDimsPerm().size() > 0) {
                indirection = packOp.getOuterDimsPerm()[i];
              }
              extractSliceOffsets.push_back(localIvs[indirection]);
            }
            for (size_t i = numLoops; i < packOp.getSourceRank(); i++) {
              extractSliceOffsets.push_back(rewriter.getIndexAttr(0));
            }

            for (int i = 0; i < numLoops; i++)
              extractSliceSizes.push_back(rewriter.getIndexAttr(1));

            for (size_t i = numLoops; i < packOp.getSourceRank(); i++) {
              extractSliceSizes.push_back(rewriter.getIndexAttr(
                  packOp.getStaticInnerTiles()[i - numLoops]));
            }
            for (size_t i = 0; i < packOp.getSourceRank(); i++)
              extractSliceStrides.push_back(rewriter.getIndexAttr(1));
          }
          auto tensorExtractType =
              tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
                  packOp.getStaticInnerTiles().size(), packOp.getSourceType(),
                  extractSliceOffsets, extractSliceSizes, extractSliceStrides);

          auto tensorExtract = rewriter.create<tensor::ExtractSliceOp>(
              loc, tensorExtractType, packOp.getSource(), extractSliceOffsets,
              extractSliceSizes, extractSliceStrides);

          auto insertSliceOp = rewriter.create<tensor::InsertSliceOp>(
              loc, tensorExtract.getResult(), iterArgs[0], insertSliceOffsets,
              insertSliceSizes, insertSliceStrides);

          return {insertSliceOp};
        });
    rewriter.replaceOp(packOp, loopNest.loops[0].getResults()[0]);
    return success();
  }
};

void populatePackUnpackOptimizationPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertPackUnpackOptimizationOp<mlir::tensor::PackOp>>(
      patterns.getContext());
  patterns.add<ConvertPackUnpackOptimizationOp<mlir::tensor::UnPackOp>>(
      patterns.getContext());
}

struct ConvertPackUnpackOptimization
    : public ConvertPackUnpackOptimizationBase<ConvertPackUnpackOptimization> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populatePackUnpackOptimizationPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createConvertPackUnpackOptimization() {
  return std::make_unique<ConvertPackUnpackOptimization>();
}
