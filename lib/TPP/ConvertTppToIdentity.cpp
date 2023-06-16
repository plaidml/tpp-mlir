#include "TPP/BuilderUtils.h"
#include "TPP/Dialect/Tpp/TppOps.h"
#include "TPP/Dialect/Tpp/TppUtils.h"
#include "TPP/Passes.h"
#include "TPP/TransformUtils.h"
#include "TPP/Transforms.h"
#include "TPP/VNNIUtils.h"
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
#include <iostream>
using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

namespace {
struct ConvertTppIdentityOp : public OpRewritePattern<tensor::PackOp> {
  using OpRewritePattern<tensor::PackOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::PackOp packOp,
                                PatternRewriter &rewriter) const override {
    if (packOp.getStaticInnerTiles().size() > 1) {
      auto constUpperBoundZero =
          getConstIndex(rewriter, packOp.getSource().getType().getShape()[0] /
                                      packOp.getStaticInnerTiles()[0]);
      auto constUpperBoundOne =
          getConstIndex(rewriter, packOp.getSource().getType().getShape()[1] /
                                      packOp.getStaticInnerTiles()[1]);
      auto zero = getConstIndex(rewriter, 0);
      auto one = getConstIndex(rewriter, 1);
      SmallVector<Value> lbs;
      lbs.push_back(zero);
      lbs.push_back(zero);
      SmallVector<Value> ubs;
      ubs.push_back(constUpperBoundZero);
      ubs.push_back(constUpperBoundOne);
      SmallVector<Value> steps;
      steps.push_back(one);
      steps.push_back(one);

      Value pos;
      Value tensorEmpty =
          packOp.getDest(); // rewriter.create<tensor::EmptyOp>(packOp.getLoc(),
                            // packOp.getDest().getType().cast<ShapedType>().getShape(),
                            // packOp.getDest().getType().getElementType());
      SmallVector<Value> reduc = {
          tensorEmpty,
      };

      auto bodyBuilder = [&](OpBuilder &builder, Location, Value iv,
                             MutableArrayRef<Value> reduc) {};

      auto loopNest = mlir::scf::buildLoopNest(
          rewriter, packOp.getLoc(), lbs, ubs, steps, reduc,
          [&reduc, &pos, bodyBuilder,
           &packOp](OpBuilder &rewriter, Location loc, ValueRange localIvs,
                    ValueRange iterArgs) -> scf::ValueVector {
            reduc.assign(iterArgs.begin(), iterArgs.end());
            Value muliOpi = rewriter.create<arith::MulIOp>(
                loc, localIvs[0],
                getConstIndex(rewriter, packOp.getStaticInnerTiles()[0]));
            SmallVector<OpFoldResult> offsets;
            offsets.push_back(muliOpi);
            Value muliOpj = rewriter.create<arith::MulIOp>(
                loc, localIvs[1],
                getConstIndex(rewriter, packOp.getStaticInnerTiles()[1]));
            offsets.push_back(muliOpj);

            SmallVector<OpFoldResult> strides;
            strides.push_back(rewriter.getIndexAttr(1));
            strides.push_back(rewriter.getIndexAttr(1));
            auto tensorExtract = rewriter.create<tensor::ExtractSliceOp>(
                loc, packOp.getSource(), offsets, packOp.getMixedTiles(),
                strides);
            auto tensorIdentityOp = rewriter.create<tpp::IdentityOp>(
                loc, tensorExtract.getResult(),
                tensorExtract.getResult().getType());

            SmallVector<OpFoldResult> insertSliceOffsets;
            insertSliceOffsets.push_back(localIvs[0]);
            insertSliceOffsets.push_back(localIvs[1]);
            insertSliceOffsets.push_back(rewriter.getIndexAttr(0));
            insertSliceOffsets.push_back(rewriter.getIndexAttr(0));

            SmallVector<OpFoldResult> insertSliceSizes;
            insertSliceSizes.push_back(rewriter.getIndexAttr(1));
            insertSliceSizes.push_back(rewriter.getIndexAttr(1));
            insertSliceSizes.push_back(
                rewriter.getIndexAttr(packOp.getStaticInnerTiles()[0]));
            insertSliceSizes.push_back(
                rewriter.getIndexAttr(packOp.getStaticInnerTiles()[1]));

            SmallVector<OpFoldResult> insertSliceStrides(
                packOp.getDestRank(), rewriter.getIndexAttr(1));
            auto insertSliceOp = rewriter.create<tensor::InsertSliceOp>(
                loc, tensorIdentityOp.getResult(0), iterArgs[0],
                insertSliceOffsets, insertSliceSizes, insertSliceStrides);
            return {insertSliceOp};
          });
      rewriter.replaceAllUsesWith(packOp.getResult(),
                                  loopNest.loops[0].getResults()[0]);
    }
  }
};

void populateTppToIdentityPatterns(RewritePatternSet &patterns) {
  // clang-format off
     patterns.add<ConvertTppIdentityOp>(patterns.getContext());
  // clang-format on
}

struct ConvertTppToIdentity
    : public ConvertTppToIdentityBase<ConvertTppToIdentity> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateTppToIdentityPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
    registry.insert<tpp::TppDialect>();
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createConvertTppToIdentityPass() {
  return std::make_unique<ConvertTppToIdentity>();
}
