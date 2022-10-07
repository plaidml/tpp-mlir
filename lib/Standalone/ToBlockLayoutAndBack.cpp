//===- ConvertToBlockLayoutAndBack.cpp ---------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/LinalgX/LinalgXOps.h"
#include "Standalone/Dialect/Tpp/TppUtils.h"
#include "Standalone/Passes.h"
#include "Standalone/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;
using namespace mlir::linalgx;

#define GEN_PASS_CLASSES
#include "Standalone/Passes.h.inc"

enum class BlockLayout {
  FORMAT_NCnc,
  FORMAT_KCck,
  FORMAT_NCHWc,
  FORMAT_KCRSck
};

static std::pair<AffineMap, AffineMap>
getMapsToBlockLayoutNCHW_to_NCHWc(int64_t dims, int64_t blockFactor,
                                  MLIRContext *ctx) {
  AffineMap outputMap = AffineMap::getMultiDimIdentityMap(dims, ctx);

  AffineExpr N, C, H, W, c;
  bindDims(ctx, N, C, H, W, c);
  AffineMap inputMap = AffineMap::get(
      dims, /*symbols=*/0, {{N}, {C * blockFactor + c}, {H}, {W}}, ctx);
  return {inputMap, outputMap};
}

static std::pair<AffineMap, AffineMap>
getMapsFromBlockLayoutNCHWc_NCHW(int64_t dims, int64_t blockFactor,
                                 MLIRContext *ctx) {
  std::pair<AffineMap, AffineMap> toBlockLayout =
      getMapsToBlockLayoutNCHW_to_NCHWc(dims, blockFactor, ctx);
  return {toBlockLayout.second, toBlockLayout.first};
}

static std::pair<AffineMap, AffineMap>
getMapsToBlockLayoutKCRS_to_KCRSck(int64_t dims, ArrayRef<int64_t> blockFactors,
                                   MLIRContext *ctx) {
  assert(blockFactors.size() == 2 &&
         "expect two blocking factors for KCRS_to_KCRSck");
  AffineMap outputMap = AffineMap::getMultiDimIdentityMap(dims, ctx);

  AffineExpr K, C, R, S, c, k;
  bindDims(ctx, K, C, R, S, c, k);
  int64_t blockFactorC = blockFactors[0];
  int64_t blockFactorK = blockFactors[1];
  AffineMap inputMap = AffineMap::get(
      dims, /*symbols=*/0,
      {{K * blockFactorK + k}, {C * blockFactorC + c}, {R}, {S}}, ctx);
  return {inputMap, outputMap};
}

// Get affine maps from NC layout to NCnc layout.
static std::pair<AffineMap, AffineMap>
getMapsToBlockLayoutNC_NCnc(int64_t dims, ArrayRef<int64_t> blockFactors,
                            MLIRContext *ctx) {
  assert(blockFactors.size() == 2 && "expect two blocking factors for NC_NCnc");
  AffineMap outputMap = AffineMap::getMultiDimIdentityMap(dims, ctx);

  AffineExpr N, C, n, c;
  bindDims(ctx, N, C, n, c);
  int64_t blockFactorOnN = blockFactors[0];
  int64_t blockFactorOnC = blockFactors[1];
  AffineMap inputMap =
      AffineMap::get(dims, /*symbols=*/0,
                     {{N * blockFactorOnN + n}, {C * blockFactorOnC + c}}, ctx);
  return {inputMap, outputMap};
}

// Get affine maps from KC to KCck layout.
static std::pair<AffineMap, AffineMap>
getMapsToBlockLayoutKC_KCck(int64_t dims, ArrayRef<int64_t> blockFactors,
                            MLIRContext *ctx) {
  assert(blockFactors.size() == 2 && "expect two blocking factors for KC_KCck");
  AffineMap outputMap = AffineMap::getMultiDimIdentityMap(dims, ctx);

  AffineExpr K, C, c, k;
  bindDims(ctx, K, C, c, k);
  int64_t blockFactorOnC = blockFactors[0];
  int64_t blockFactorOnK = blockFactors[1];
  AffineMap inputMap =
      AffineMap::get(dims, /*symbols=*/0,
                     {{C * blockFactorOnC + c}, {K * blockFactorOnK + k}}, ctx);
  return {inputMap, outputMap};
}

// Get affine maps fron NCnc layout to NC.
static std::pair<AffineMap, AffineMap>
getMapsFromBlockLayoutNCnc_NC(int64_t dims, ArrayRef<int64_t> blockFactors,
                              MLIRContext *ctx) {
  std::pair<AffineMap, AffineMap> toBlockLayout =
      getMapsToBlockLayoutNC_NCnc(dims, blockFactors, ctx);
  return {toBlockLayout.second, toBlockLayout.first};
}

// Given a tensor `tensor` a layout format and a block factor, relayout
// the tensor using the specified block factor and tensor relayout.
// Fail if the blocking factors do not perfectly divide the tensor
// dimension.
static FailureOr<Value> getReshapedTensor(Location loc, Value tensor,
                                          BlockLayout layout,
                                          ArrayRef<int64_t> blockFactors,
                                          RewriterBase &rewriter) {
  MLIRContext *ctx = rewriter.getContext();
  RankedTensorType unBlockedTensorType =
      tensor.getType().cast<RankedTensorType>();
  if (!unBlockedTensorType.hasStaticShape())
    return failure();
  ArrayRef<int64_t> shape = unBlockedTensorType.getShape();

  std::pair<AffineMap, AffineMap> maps;
  SmallVector<int64_t> shapeBlockedTensor;
  if (layout == BlockLayout::FORMAT_NCnc) {
    assert(blockFactors.size() == 2 &&
           "FORMAT_NCnc requires 2 blocking factor");
    assert(shape.size() == 2 && "FORMAT_NCnc requires 2d tensor");
    int64_t blockFactorOnN = blockFactors[0];
    int64_t blockFactorOnC = blockFactors[1];
    if ((shape[0] % blockFactorOnN != 0) || (shape[1] % blockFactorOnC != 0))
      return failure();
    int64_t N = shape[0] / blockFactorOnN;
    int64_t C = shape[1] / blockFactorOnC;
    int64_t n = blockFactorOnN, c = blockFactorOnC;
    shapeBlockedTensor = {N, C, n, c};
    maps = getMapsToBlockLayoutNC_NCnc(shapeBlockedTensor.size(), blockFactors,
                                       ctx);
  } else if (layout == BlockLayout::FORMAT_KCck) {
    assert(blockFactors.size() == 2 &&
           "FORMAT_KCck requires 2 blocking factor");
    assert(shape.size() == 2 && "FORMAT_KCck requires 2d tensor");
    int64_t blockFactorOnC = blockFactors[0];
    int64_t blockFactorOnK = blockFactors[1];
    if ((shape[0] % blockFactorOnC != 0) || (shape[1] % blockFactorOnK != 0))
      return failure();
    int64_t K = shape[1] / blockFactorOnK;
    int64_t C = shape[0] / blockFactorOnC;
    int64_t k = blockFactorOnK, c = blockFactorOnC;
    shapeBlockedTensor = {K, C, c, k};
    maps = getMapsToBlockLayoutKC_KCck(shapeBlockedTensor.size(), blockFactors,
                                       ctx);
  } else if (layout == BlockLayout::FORMAT_NCHWc) {
    assert(blockFactors.size() == 1 &&
           "FORMAT_NCHWc requires 1 blocking factor");
    assert(shape.size() == 4 && "FORMAT_NCHWc requires 4d tensor");
    int64_t blockFactor = blockFactors[0];
    if (shape[1] % blockFactor != 0)
      return failure();
    int64_t N = shape[0];
    int64_t C = shape[1] / blockFactor;
    int64_t H = shape[2];
    int64_t W = shape[3];
    int64_t c = blockFactor;
    shapeBlockedTensor = {N, C, H, W, c};
    maps = getMapsToBlockLayoutNCHW_to_NCHWc(shapeBlockedTensor.size(),
                                             blockFactor, ctx);
  } else {
    assert(layout == BlockLayout::FORMAT_KCRSck && "expect FORMAT_KCRSck");
    assert(blockFactors.size() == 2 &&
           "FORMAT_KCRSck requires 2 blocking factors");
    assert(shape.size() == 4 && "FORMAT_KCRSck requires 4d tensor");
    int64_t blockFactorOnC = blockFactors[0];
    int64_t blockFactorOnK = blockFactors[1];
    if ((shape[0] % blockFactorOnK != 0) || (shape[1] % blockFactorOnC != 0))
      return failure();
    int64_t K = shape[0] / blockFactorOnK;
    int64_t C = shape[1] / blockFactorOnC;
    int64_t R = shape[2];
    int64_t S = shape[3];
    int64_t c = blockFactorOnC;
    int64_t k = blockFactorOnK;
    shapeBlockedTensor = {K, C, R, S, c, k};
    maps = getMapsToBlockLayoutKCRS_to_KCRSck(shapeBlockedTensor.size(),
                                              blockFactors, ctx);
  }
  Value emptyTensor = rewriter.create<tensor::EmptyOp>(
      loc, shapeBlockedTensor, unBlockedTensorType.getElementType());
  return rewriter
      .create<linalgx::Relayout>(loc, emptyTensor.getType(), tensor, emptyTensor,
                                 maps.first, maps.second)
      .getResult()[0];
}

static LogicalResult BlockOpPreconditions(linalg::LinalgOp linalgOp) {
  if (linalgOp.hasDynamicShape() || linalgOp.hasBufferSemantics())
    return failure();
  return success();
}

FailureOr<linalg::GenericOp>
mlir::linalgx::blockConv2DNchwFchwOp(RewriterBase &rewriter,
                                     linalg::Conv2DNchwFchwOp convOp,
                                     ArrayRef<int64_t> blockingFactors) {
  if (blockingFactors.size() != 2)
    return failure();
  if (failed(BlockOpPreconditions(convOp)))
    return failure();

  Location loc = convOp.getLoc();
  MLIRContext *ctx = convOp.getContext();
  SmallVector<Value, 2> reshapedInputTensors;

  SmallVector<Value> inputOperands = convOp.getInputOperands();
  SmallVector<Value> outputOperands = convOp.getOutputOperands();

  // reshape the img and the filter.
  int64_t blockingFactorOnC = blockingFactors[0];
  int64_t blockingFactorOnK = blockingFactors[1];
  SmallVector<int64_t> currentBlockingFactors = {blockingFactorOnC};
  BlockLayout blockLayout = BlockLayout::FORMAT_NCHWc;
  for (Value input : inputOperands) {
    FailureOr<Value> maybeReshaped = getReshapedTensor(
        loc, input, blockLayout, currentBlockingFactors, rewriter);
    if (failed(maybeReshaped))
      return failure();
    reshapedInputTensors.push_back(*maybeReshaped);
    currentBlockingFactors = {blockingFactorOnC, blockingFactorOnK};
    blockLayout = BlockLayout::FORMAT_KCRSck;
  }

  blockLayout = BlockLayout::FORMAT_NCHWc;
  currentBlockingFactors = {blockingFactorOnC};
  FailureOr<Value> maybeReshapedOutputTensor = getReshapedTensor(
      loc, outputOperands[0], blockLayout, currentBlockingFactors, rewriter);
  if (failed(maybeReshapedOutputTensor))
    return failure();
  Value reshapedOutputTensor = *maybeReshapedOutputTensor;

  // Swap conv with generic.
  //         N   K   P   Q   k   C   R   S   c
  AffineExpr p1, p2, p3, p4, p5, r1, r2, r3, r4;
  bindDims(ctx, p1, p2, p3, p4, p5, r1, r2, r3, r4);
  AffineMap mapOut =
      AffineMap::get(/*dims=*/9, /*symbols=*/0, {p1, p2, p3, p4, p5}, ctx);
  AffineMap mapImg = AffineMap::get(/*dims=*/9, /*symbols=*/0,
                                    {p1, r1, p3 + r2, p4 + r3, r4}, ctx);
  AffineMap mapFil =
      AffineMap::get(/*dims=*/9, /*symbols=*/0, {p2, r1, r2, r3, r4, p5}, ctx);
  linalg::GenericOp replacementOp = rewriter.create<linalg::GenericOp>(
      loc, reshapedOutputTensor.getType(), reshapedInputTensors,
      ValueRange{reshapedOutputTensor},
      ArrayRef<AffineMap>{mapImg, mapFil, mapOut},
      ArrayRef<StringRef>{
          getParallelIteratorTypeName(), getParallelIteratorTypeName(),
          getParallelIteratorTypeName(), getParallelIteratorTypeName(),
          getParallelIteratorTypeName(), getReductionIteratorTypeName(),
          getReductionIteratorTypeName(), getReductionIteratorTypeName(),
          getReductionIteratorTypeName()},
      /*doc=*/"", /*libraryCall=*/"tpp.blocked.Conv2DNchwFchwOp");
  rewriter.inlineRegionBefore(convOp->getRegion(0), replacementOp.getRegion(),
                              replacementOp.getRegion().begin());

  // convert back from block layout.
  assert(currentBlockingFactors.size() == 1);
  Value outBlockedTensor = replacementOp.getResult(0);
  Value outUnBlockedTensor = outputOperands[0];
  std::pair<AffineMap, AffineMap> maps = getMapsFromBlockLayoutNCHWc_NCHW(
      outBlockedTensor.getType().cast<ShapedType>().getRank(),
      currentBlockingFactors[0], ctx);
  Value outReplacement =
      rewriter
          .create<linalgx::Relayout>(loc, outUnBlockedTensor.getType(),
                                     outBlockedTensor, outUnBlockedTensor,
                                     maps.first, maps.second)
          .getResults()[0];
  rewriter.replaceOp(convOp, outReplacement);
  return replacementOp;
}

FailureOr<linalg::GenericOp>
mlir::linalgx::blockMatmulOp(RewriterBase &rewriter, linalg::MatmulOp matmulOp,
                             ArrayRef<int64_t> blockingFactors) {
  if (blockingFactors.size() != 2)
    return failure();
  if (failed(BlockOpPreconditions(matmulOp)))
    return failure();

  Location loc = matmulOp.getLoc();
  MLIRContext *ctx = matmulOp.getContext();
  SmallVector<Value, 2> reshapedInputTensors;
  // reshape input A to NCnc and B to KCck.
  BlockLayout blockLayout = BlockLayout::FORMAT_NCnc;
  for (Value input : matmulOp.getInputs()) {
    FailureOr<Value> maybeReshaped =
        getReshapedTensor(loc, input, blockLayout, blockingFactors, rewriter);
    if (failed(maybeReshaped))
      return failure();
    reshapedInputTensors.push_back(*maybeReshaped);
    blockLayout = BlockLayout::FORMAT_KCck;
  }

  blockLayout = BlockLayout::FORMAT_NCnc;
  FailureOr<Value> maybeReshapedOutputTensor = getReshapedTensor(
      loc, matmulOp.getOutputs()[0], blockLayout, blockingFactors, rewriter);
  if (failed(maybeReshapedOutputTensor))
    return failure();
  Value reshapedOutputTensor = *maybeReshapedOutputTensor;

  // swap linalg.matmul with a linalg.generic.
  AffineExpr p1, p2, r1, p3, p4, r2;
  bindDims(ctx, p1, p2, r1, p3, p4, r2);
  AffineMap mapA =
      AffineMap::get(/*dims=*/6, /*symbols=*/0, {p1, r1, p3, r2}, ctx);
  AffineMap mapB =
      AffineMap::get(/*dims=*/6, /*symbols=*/0, {p2, r1, r2, p4}, ctx);
  AffineMap mapC =
      AffineMap::get(/*dims=*/6, /*symbols=*/0, {p1, p2, p3, p4}, ctx);
  linalg::GenericOp replacementOp = rewriter.create<linalg::GenericOp>(
      loc, reshapedOutputTensor.getType(), reshapedInputTensors,
      ValueRange{reshapedOutputTensor}, ArrayRef<AffineMap>{mapA, mapB, mapC},
      ArrayRef<StringRef>{
          getParallelIteratorTypeName(), getParallelIteratorTypeName(),
          getReductionIteratorTypeName(), getParallelIteratorTypeName(),
          getParallelIteratorTypeName(), getReductionIteratorTypeName()},
      /*doc=*/"", /*libraryCall=*/"");
  rewriter.inlineRegionBefore(matmulOp.getRegion(), replacementOp.getRegion(),
                              replacementOp.getRegion().begin());

  // convert back from block layout.
  Value outBlockedTensor = replacementOp.getResult(0);
  Value outUnBlockedTensor = matmulOp.getOutputs()[0];
  std::pair<AffineMap, AffineMap> maps = getMapsFromBlockLayoutNCnc_NC(
      outBlockedTensor.getType().cast<ShapedType>().getRank(), blockingFactors,
      ctx);
  Value outReplacement =
      rewriter
          .create<linalgx::Relayout>(loc, outUnBlockedTensor.getType(),
                                     outBlockedTensor, outUnBlockedTensor,
                                     maps.first, maps.second)
          .getResults()[0];
  rewriter.replaceOp(matmulOp, outReplacement);
  return replacementOp;
}

namespace {

// Relayout MatmulOp as following:
// [NB][KB][nb][kb] += [NB][CB][nb][cb] * [KB][CB][cb][kb]
// CB = batch reduce dimension.
struct DoItOnMatmul : public OpRewritePattern<linalg::MatmulOp> {
  DoItOnMatmul(MLIRContext *context, ArrayRef<int64_t> blockingFactors,
               PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::MatmulOp>(context, benefit),
        blockingFactors(blockingFactors) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    FailureOr<linalg::GenericOp> blockedMatmul =
        mlir::linalgx::blockMatmulOp(rewriter, matmulOp, blockingFactors);
    if (failed(blockedMatmul))
      return failure();
    return success();
  }

private:
  ArrayRef<int64_t> blockingFactors;
};

// TODO: generalize to any elementWise operation.
// Relayout relu as: [NB][KB][nb][kb]
struct SinkBlockLayoutAfterRelu : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  bool isInPlaceRelu(linalg::GenericOp linalgOp) const {
    return linalgOp.getNumInputs() == 0;
  }

  // Note this is not what you want. The blocking should be part of the
  // operation as we did in IREE, not hidden in the affine map. Attempt to
  // recover the constants (aka blocks factor) in the map, assert on failure.
  SmallVector<int64_t> getBlockingFactors(linalgx::Relayout relayoutOp) const {

    // Walk the affine expression and return the first constant if any.
    // Assert if we don't find any constant or multiple constants are found.
    auto getConstant = [&](AffineExpr expr) -> int64_t {
      bool onlyOneConstant = false;
      int64_t value;
      expr.walk([&](AffineExpr e) -> void {
        if (e.getKind() == AffineExprKind::Constant) {
          assert(!onlyOneConstant &&
                 "failed to extract constant from relayout");
          value = e.cast<AffineConstantExpr>().getValue();
          onlyOneConstant = true;
        }
      });
      assert(onlyOneConstant == true &&
             "failed to extract constant from relayout");
      return value;
    };

    AffineMap outputMap = relayoutOp.getOutputMap();
    SmallVector<int64_t> blocks;
    blocks.reserve(outputMap.getNumResults());
    ArrayRef<AffineExpr> results = outputMap.getResults();
    for (AffineExpr expr : results)
      blocks.push_back(getConstant(expr));
    return blocks;
  }

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!tpp::isMarkedWithTpp(linalgOp, "tpp.relu"))
      return failure();
    Location loc = linalgOp.getLoc();

    Value operand = (isInPlaceRelu(linalgOp))
                        ? linalgOp.getOutputOperand(0)->get()
                        : linalgOp.getInputOperand(0)->get();

    linalgx::Relayout fromBlockLayout =
        operand.getDefiningOp<linalgx::Relayout>();
    if (!fromBlockLayout || !fromBlockLayout->getResult(0).hasOneUse())
      return failure();

    Value blockTensor = fromBlockLayout.getInputs()[0];
    ShapedType blockTensorType = blockTensor.getType().cast<ShapedType>();
    BlockLayout blockLayout = BlockLayout::FORMAT_NCnc;
    FailureOr<Value> maybeReluBuffer =
        getReshapedTensor(loc, linalgOp.getOutputs()[0], blockLayout,
                          getBlockingFactors(fromBlockLayout), rewriter);
    if (failed(maybeReluBuffer))
      return failure();
    Value reluBuffer = *maybeReluBuffer;

    AffineMap mapI =
        AffineMap::getMultiDimIdentityMap(/*dims=*/4, linalgOp.getContext());
    AffineMap mapO = mapI;
    linalg::GenericOp newReluOp =
        (isInPlaceRelu(linalgOp))
            ? rewriter.create<linalg::GenericOp>(
                  loc, blockTensorType, llvm::None, ValueRange{blockTensor},
                  ArrayRef<AffineMap>{mapO},
                  ArrayRef<StringRef>{getParallelIteratorTypeName(),
                                      getParallelIteratorTypeName(),
                                      getParallelIteratorTypeName(),
                                      getParallelIteratorTypeName()},
                  /*doc=*/"", /*libraryCall=*/"tpp.relu")
            : rewriter.create<linalg::GenericOp>(
                  loc, blockTensorType, ValueRange{blockTensor},
                  ValueRange{reluBuffer}, ArrayRef<AffineMap>{mapI, mapO},
                  ArrayRef<StringRef>{getParallelIteratorTypeName(),
                                      getParallelIteratorTypeName(),
                                      getParallelIteratorTypeName(),
                                      getParallelIteratorTypeName()},
                  /*doc=*/"", /*libraryCall=*/"tpp.relu");
    rewriter.inlineRegionBefore(linalgOp.getRegion(), newReluOp.getRegion(),
                                newReluOp.getRegion().begin());

    Value outUnBlockedTensor = (isInPlaceRelu(linalgOp))
                                   ? fromBlockLayout.getOutput()
                                   : linalgOp.getOutputOperand(0)->get();

    Type outUnBlockedTensorType = outUnBlockedTensor.getType();
    std::pair<AffineMap, AffineMap> maps = getMapsFromBlockLayoutNCnc_NC(
        /*dims=*/4, getBlockingFactors(fromBlockLayout), linalgOp.getContext());

    Value outReplacement =
        rewriter
            .create<linalgx::Relayout>(
                loc, outUnBlockedTensorType, newReluOp->getResult(0),
                outUnBlockedTensor, maps.first, maps.second)
            .getResult()[0];
    rewriter.replaceOp(linalgOp, outReplacement);
    return success();
  }
};

// TODO: should be part of a more structured de-generalization pass.
// pattern to go from linalg.generic {tpp.matmul} to linalg.matmul.
// see:
// https://sourcegraph.com/github.com/iree-org/iree/-/blob/llvm-external-projects/iree-dialects/lib/Dialect/LinalgTransform/IR/StructuredTransformOpsExt.cpp?L124
struct DeGeneralizeMatmul : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!linalgOp.hasTensorSemantics())
      return failure();
    if (!tpp::isMarkedWithTpp(linalgOp, "tpp.matmul"))
      return failure();
    SmallVector<Value> inputOperands = linalgOp.getInputOperands();
    SmallVector<Value> outputOperands = linalgOp.getOutputOperands();
    rewriter.replaceOpWithNewOp<linalg::MatmulOp>(
        linalgOp, linalgOp.getResultTypes(), inputOperands, outputOperands);
    return success();
  }
};

struct BlockMatmulLayout : public BlockMatmulLayoutBase<BlockMatmulLayout> {
  BlockMatmulLayout() = default;
  BlockMatmulLayout(ArrayRef<int64_t> blockingFactors) {
    this->blockingFactors = blockingFactors;
  }

  void runOnOperation() override {
    if (blockingFactors.empty())
      return;
    MLIRContext *ctx = getOperation().getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<DoItOnMatmul>(ctx, blockingFactors);
    patterns.add<DeGeneralizeMatmul, SinkBlockLayoutAfterRelu>(ctx);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

struct DoItOnConv2DNchwFchw
    : public OpRewritePattern<linalg::Conv2DNchwFchwOp> {
  DoItOnConv2DNchwFchw(MLIRContext *context, ArrayRef<int64_t> blockingFactors,
                       PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::Conv2DNchwFchwOp>(context, benefit),
        blockingFactors(blockingFactors) {}

  LogicalResult matchAndRewrite(linalg::Conv2DNchwFchwOp linalgOp,
                                PatternRewriter &rewriter) const override {
    FailureOr<linalg::GenericOp> maybeGeneric =
        mlir::linalgx::blockConv2DNchwFchwOp(rewriter, linalgOp,
                                             blockingFactors);
    if (failed(maybeGeneric))
      return failure();
    return success();
  }

private:
  ArrayRef<int64_t> blockingFactors;
};

struct BlockConv2DNchwFchwLayout
    : public BlockConv2DNchwFchwLayoutBase<BlockConv2DNchwFchwLayout> {
  BlockConv2DNchwFchwLayout() = default;
  BlockConv2DNchwFchwLayout(ArrayRef<int64_t> blockingFactors) {
    this->blockingFactors = blockingFactors;
  }

  void runOnOperation() override {
    if (blockingFactors.empty())
      return;
    MLIRContext *ctx = getOperation().getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<DoItOnConv2DNchwFchw>(ctx, blockingFactors);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // end namespace

void mlir::tpp::populateSinkRelayoutPatterns(RewritePatternSet &patterns) {
  // clang-format off
  mlir::tpp::populateMapLinalgToTppPatterns(patterns);
  patterns.add<SinkBlockLayoutAfterRelu>(patterns.getContext());
  // clang-format on
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createBlockMatmulLayout() {
  return std::make_unique<BlockMatmulLayout>();
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createBlockConv2DNchwFchwLayout() {
  return std::make_unique<BlockConv2DNchwFchwLayout>();
}
