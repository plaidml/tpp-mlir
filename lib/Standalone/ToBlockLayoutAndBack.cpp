//===- ConvertToBlockLayoutAndBack.cpp ---------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/LinalgX/LinalgXOps.h"
#include "Standalone/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::linalgx;

#define GEN_PASS_CLASSES
#include "Standalone/Passes.h.inc"

namespace {

struct DoItOnMatmul : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  Value getReshapedTensor(Location loc, Value tensor, int64_t blockFactor,
                          PatternRewriter &rewriter) const {
    RankedTensorType unBlockedTensorType =
        tensor.getType().cast<RankedTensorType>();
    ArrayRef<int64_t> shape = unBlockedTensorType.getShape();
    SmallVector<int64_t> shapeBlockedTensor = {shape[0] / blockFactor,
                                               shape[1] / blockFactor,
                                               blockFactor, blockFactor};
    Value initOp = rewriter.create<linalg::InitTensorOp>(
        loc, shapeBlockedTensor, unBlockedTensorType.getElementType());
    return initOp;
  }

  std::pair<AffineMap, AffineMap>
  getToBlockLayout(int64_t dims, int64_t blockFactor, MLIRContext *ctx) const {
    AffineMap outputMap = AffineMap::getMultiDimIdentityMap(dims, ctx);

    AffineExpr N, C, n, c;
    bindDims(ctx, N, C, n, c);
    AffineMap inputMap =
        AffineMap::get(dims, /*symbols*/ 0,
                       {{N * blockFactor + n}, {C * blockFactor + c}}, ctx);
    return {inputMap, outputMap};
  }

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    if (matmulOp.hasDynamicShape() || matmulOp.hasBufferSemantics())
      return failure();

    Location loc = matmulOp.getLoc();
    MLIRContext *ctx = matmulOp.getContext();
    int64_t blockFactor = 32;
    SmallVector<Value, 2> reshapedInputTensors;
    // reshape input.
    for (Value input : matmulOp.getInputs()) {
      Value initOp = getReshapedTensor(loc, input, blockFactor, rewriter);
      std::pair<AffineMap, AffineMap> maps = getToBlockLayout(
          initOp.getType().cast<ShapedType>().getRank(), blockFactor, ctx);
      Value reshapedInputTensor =
          rewriter
              .create<linalgx::ToBlockLayout>(loc, initOp.getType(), input,
                                              initOp, maps.first, maps.second)
              .getOutput();
      reshapedInputTensors.push_back(reshapedInputTensor);
    }

    Value reshapedOutputTensor = nullptr;
    for (Value output : matmulOp.getOutputs())
      reshapedOutputTensor =
          getReshapedTensor(loc, output, blockFactor, rewriter);

    // swap linalg.matmul with a linalg.generic.
    AffineExpr p1, p2, r1, p3, p4, r2;
    bindDims(ctx, p1, p2, r1, p3, p4, r2);
    AffineMap mapA =
        AffineMap::get(/*dims*/ 6, /*symbols*/ 0, {p1, r1, p3, r2}, ctx);
    AffineMap mapB =
        AffineMap::get(/*dims*/ 6, /*symbols*/ 0, {r1, p2, r2, p4}, ctx);
    AffineMap mapC =
        AffineMap::get(/*dims*/ 6, /*symbols*/ 0, {p1, p2, p3, p4}, ctx);
    linalg::GenericOp replacementOp = rewriter.create<linalg::GenericOp>(
        loc, reshapedOutputTensor.getType(), reshapedInputTensors,
        ValueRange{reshapedOutputTensor}, ArrayRef<AffineMap>{mapA, mapB, mapC},
        ArrayRef<StringRef>{
            getParallelIteratorTypeName(), getParallelIteratorTypeName(),
            getReductionIteratorTypeName(), getParallelIteratorTypeName(),
            getParallelIteratorTypeName(), getReductionIteratorTypeName()},
        /*doc=*/"", /*libraryCall=*/"");
    rewriter.inlineRegionBefore(matmulOp.region(), replacementOp.region(),
                                replacementOp.region().begin());

    // convert back from block layout.
    Value outBlockedTensor = replacementOp.getResult(0);
    Value outUnBlockedTensor = matmulOp.getOutputs()[0];
    std::pair<AffineMap, AffineMap> maps = getToBlockLayout(
        outBlockedTensor.getType().cast<ShapedType>().getRank(), blockFactor,
        ctx);
    Value outReplacement =
        rewriter
            .create<linalgx::FromBlockLayout>(
                loc, outUnBlockedTensor.getType(), outBlockedTensor,
                outUnBlockedTensor, maps.second, maps.first)
            .getResults()[0];
    rewriter.replaceOp(matmulOp, outReplacement);
    return success();
  }
};

struct SinkBlockLayoutAfterRelu : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    std::string libraryCall = linalgOp.getLibraryCallName();
    if (libraryCall.compare("tpp.relu") != 0)
      return failure();
    if (linalgOp.getNumInputs() == 0)
      return failure();
    Location loc = linalgOp.getLoc();
    Value operand = linalgOp.getInputOperand(0)->get();
    linalgx::FromBlockLayout fromBlockLayout =
        operand.getDefiningOp<linalgx::FromBlockLayout>();
    if (!fromBlockLayout || !fromBlockLayout.outputs()[0].hasOneUse())
      return failure();

    Value reluBuffer = rewriter.create<linalg::InitTensorOp>(
        loc,
        fromBlockLayout.inputs()[0].getType().cast<ShapedType>().getShape(),
        fromBlockLayout.inputs()[0]
            .getType()
            .cast<ShapedType>()
            .getElementType());

    return failure();
  }
};

struct ToBlockLayoutAndBack
    : public ToBlockLayoutAndBackBase<ToBlockLayoutAndBack> {
  void runOnOperation() override {
    MLIRContext *ctx = getOperation().getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<DoItOnMatmul, SinkBlockLayoutAfterRelu>(ctx);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // end namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createToBlockLayoutAndBackPass() {
  return std::make_unique<ToBlockLayoutAndBack>();
}
