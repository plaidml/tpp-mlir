//===- DecomposeConvToMatmul.cpp ---------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "Standalone/Passes.h.inc"

namespace {

struct DecomposeConv : OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp convOp,
                                PatternRewriter &rewriter) const override {

    // do not handle convolutions with dilation and strides.
    if (DenseIntElementsAttr dilations = convOp.dilations()) {
      auto values = dilations.getValues<APInt>();
      if (llvm::any_of(values, [](const APInt &value) {
            return value.getSExtValue() != 1;
          })) {
        return failure();
      }
    }
    if (DenseIntElementsAttr strides = convOp.strides()) {
      auto values = strides.getValues<APInt>();
      if (llvm::any_of(values, [](const APInt &value) {
            return value.getSExtValue() != 1;
          })) {
        return failure();
      }
    }

    Value image = convOp.image();
    Value filter = convOp.filter();
    Value output = convOp.outputs()[0];

    ShapedType imageType = image.getType().cast<ShapedType>();
    ShapedType filterType = filter.getType().cast<ShapedType>();
    ShapedType outputType = output.getType().cast<ShapedType>();

    // only static dimensions.
    if ((!imageType.hasStaticShape()) || (!filterType.hasStaticShape()) ||
        (!outputType.hasStaticShape()))
      return failure();

    return failure();
  }
};

void populateConvDecomposePatterns(RewritePatternSet &patterns) {
  patterns.insert<DecomposeConv>(patterns.getContext());
}

struct DecomposeConvToMatmul
    : public DecomposeConvToMatmulBase<DecomposeConvToMatmul> {
  void runOnOperation() override {
    RewritePatternSet patterns(getOperation().getContext());
    populateConvDecomposePatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // end namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createDecomposeConvToMatmulPass() {
  return std::make_unique<DecomposeConvToMatmul>();
}
