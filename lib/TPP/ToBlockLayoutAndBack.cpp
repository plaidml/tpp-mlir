//===- ConvertToBlockLayoutAndBack.cpp ---------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/LinalgX/LinalgXOps.h"
#include "TPP/Dialect/Tpp/TppUtils.h"
#include "TPP/Passes.h"
#include "TPP/Transforms.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::linalgx;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

/// Helper function to create the pack operation.
static Value toPackLayoutImpl(Location loc, Value input,
                              ArrayRef<OpFoldResult> tiles,
                              ArrayRef<int64_t> innerDimsPos,
                              ArrayRef<int64_t> outerDimsPerm,
                              OpBuilder &builder, bool useAlloc = false) {
  SmallVector<Value> dynamicTiles;
  SmallVector<int64_t> staticTiles;
  dispatchIndexOpFoldResults(tiles, dynamicTiles, staticTiles,
                             ShapedType::kDynamicSize);
  ShapedType result = PackOp::getPackedType(input.getType(), staticTiles,
                                            innerDimsPos, outerDimsPerm);
  ShapedType inputType = input.getType().cast<ShapedType>();
  ArrayRef<int64_t> shape = result.getShape();
  Value output;
  if (useAlloc)
    output = builder.create<bufferization::AllocTensorOp>(
        loc, RankedTensorType::get(shape, inputType.getElementType()),
        ValueRange{});
  else
    output =
        builder.create<tensor::EmptyOp>(loc, shape, inputType.getElementType());
  return builder
      .create<linalgx::PackOp>(loc, input, output, innerDimsPos, outerDimsPerm,
                               tiles)
      .getResults()[0];
}

/// Helper function to create the unpack operation.
static Value toUnPackLayoutImpl(Location loc, Value input, Value output,
                                ArrayRef<OpFoldResult> tiles,
                                ArrayRef<int64_t> innerDimPos,
                                ArrayRef<int64_t> outerDimsPerm,
                                OpBuilder &builder) {
  return builder
      .create<linalgx::UnPackOp>(loc, input, output, innerDimPos, outerDimsPerm,
                                 tiles)
      .getResults()[0];
}

static Value handleLayoutNC_NCnc(Location loc, Value input, Value output,
                                 ArrayRef<OpFoldResult> tiles,
                                 OpBuilder &builder, bool useAlloc = false) {
  assert(tiles.size() == 2 && "expect two tile sizes for NC_NCnc");
  SmallVector<int64_t> innerDimPos = {0, 1};
  if (!output)
    return toPackLayoutImpl(loc, input, tiles, innerDimPos, {}, builder,
                            useAlloc);
  return toUnPackLayoutImpl(loc, input, output, tiles, innerDimPos, {},
                            builder);
}

/// Helper function to pack from NC to NCnc.
static Value toPackLayoutNC_NCnc(Location loc, Value input,
                                 ArrayRef<OpFoldResult> tiles,
                                 OpBuilder &builder, bool useAlloc = false) {
  return handleLayoutNC_NCnc(loc, input, nullptr, tiles, builder, useAlloc);
}

/// Helper function to unpack from NCnc to NC.
static Value fromPackLayoutNCnc_NC(Location loc, Value input, Value output,
                                   ArrayRef<OpFoldResult> tiles,
                                   OpBuilder &builder) {
  return handleLayoutNC_NCnc(loc, input, output, tiles, builder);
}

static Value handleLayoutNCHW_NCHWc(Location loc, Value input, Value output,
                                    ArrayRef<OpFoldResult> tiles,
                                    OpBuilder &builder, bool useAlloc = false) {
  assert(tiles.size() == 1 && "expect one tile size for NCHW_NCHWc");
  SmallVector<int64_t> innerDimPos = {1};
  if (!output)
    return toPackLayoutImpl(loc, input, tiles, innerDimPos, {}, builder,
                            useAlloc);
  return toUnPackLayoutImpl(loc, input, output, tiles, innerDimPos, {},
                            builder);
}

/// Helper function to pack from NCHW to NCHWc.
static Value toPackLayoutNCHW_NCHWc(Location loc, Value input,
                                    ArrayRef<OpFoldResult> tiles,
                                    OpBuilder &builder, bool useAlloc = false) {
  return handleLayoutNCHW_NCHWc(loc, input, nullptr, tiles, builder, useAlloc);
}

/// Helper function to unpack from NCHWc to NCHW.
static Value fromPackLayoutNCHWc_NCHW(Location loc, Value input, Value output,
                                      ArrayRef<OpFoldResult> tiles,
                                      OpBuilder &builder) {
  return handleLayoutNCHW_NCHWc(loc, input, output, tiles, builder);
}

/// Helper function to pack from KC to CKkc.
static Value toPackLayoutKC_CKkc(Location loc, Value input,
                                 ArrayRef<OpFoldResult> tiles,
                                 OpBuilder &builder, bool useAlloc = false) {
  assert(tiles.size() == 2 && "expect two tiles size for KC_CKkc");
  SmallVector<int64_t> innerDimPos = {0, 1};
  SmallVector<int64_t> outerDimPerm = {1, 0};
  return toPackLayoutImpl(loc, input, tiles, innerDimPos, outerDimPerm, builder,
                          useAlloc);
}

static Value handleLayoutNPQK_NKPQk(Location loc, Value input, Value output,
                                    ArrayRef<OpFoldResult> tiles,
                                    OpBuilder &builder, bool useAlloc = false) {
  assert(tiles.size() == 1 && "expect one tile size for NPQK_NKPQk");
  SmallVector<int64_t> innerDimsPos = {3};
  SmallVector<int64_t> outerDimsPerm = {0, 3, 1, 2};
  if (!output)
    return toPackLayoutImpl(loc, input, tiles, innerDimsPos, outerDimsPerm,
                            builder, useAlloc);
  return toUnPackLayoutImpl(loc, input, output, tiles, innerDimsPos,
                            outerDimsPerm, builder);
}

/// Helper function to pack NPQK to NKPQk.
static Value toPackLayoutNPQK_NKPQk(Location loc, Value input,
                                    ArrayRef<OpFoldResult> tiles,
                                    OpBuilder &builder, bool useAlloc = false) {
  return handleLayoutNPQK_NKPQk(loc, input, nullptr, tiles, builder, useAlloc);
}

/// Helper function to unpack NKPQk to NPQK.
static Value fromPackLayoutNKPQk_NPQK(Location loc, Value input, Value output,
                                      ArrayRef<OpFoldResult> tiles,
                                      OpBuilder &builder) {
  return handleLayoutNPQK_NKPQk(loc, input, output, tiles, builder);
}

/// Helper function to pack from RSCK to KCRSck.
static Value toPackLayoutRSCK_KCRSck(Location loc, Value input,
                                     ArrayRef<OpFoldResult> tiles,
                                     OpBuilder &builder) {
  assert(tiles.size() == 2 && "expect two tiles for RSCK_KCRSck");
  SmallVector<int64_t> innerDimsPos = {2, 3};
  SmallVector<int64_t> outerDimsPerm = {3, 2, 0, 1};
  return toPackLayoutImpl(loc, input, tiles, innerDimsPos, outerDimsPerm,
                          builder);
}

/// Helper function to pack from KCRS to KCRSck.
static Value toPackLayoutKCRS_KCRSck(Location loc, Value input,
                                     ArrayRef<OpFoldResult> tiles,
                                     OpBuilder &builder,
                                     bool useAlloc = false) {
  assert(tiles.size() == 2 && "expect two tiles size for KCRS_KCRSck");
  SmallVector<int64_t> innerDimPos = {1, 0};
  return toPackLayoutImpl(loc, input, tiles, innerDimPos, {}, builder,
                          useAlloc);
}

template <typename OpTy>
static FailureOr<linalg::GenericOp>
packConvolutions(RewriterBase &rewriter, OpTy convOp,
                 ArrayRef<OpFoldResult> tiles) {
  static_assert(llvm::is_one_of<OpTy, linalg::Conv2DNhwcHwcfOp,
                                linalg::Conv2DNchwFchwOp>::value,
                "applies to only pack or unpack operations");

  if (tiles.size() != 2)
    return rewriter.notifyMatchFailure(convOp, "require 2 tile factors");
  if (convOp.hasDynamicShape())
    return rewriter.notifyMatchFailure(convOp, "require static shape");
  if (convOp.hasBufferSemantics())
    return rewriter.notifyMatchFailure(convOp, "require tensor semantics");

  bool isConv2DNhwcHwcfOp =
      (std::is_same<OpTy, linalg::Conv2DNhwcHwcfOp>::value) ? true : false;

  Location loc = convOp.getLoc();
  MLIRContext *ctx = convOp.getContext();

  SmallVector<Value> inputOperands = convOp.getDpsInputOperands();
  SmallVector<Value> outputOperands = convOp.getDpsInitOperands();

  // pack the image and the filter.
  Value image = inputOperands[0];
  Value packedImage =
      (isConv2DNhwcHwcfOp)
          ? toPackLayoutNPQK_NKPQk(loc, image, tiles[0], rewriter)
          : toPackLayoutNCHW_NCHWc(loc, image, tiles[0], rewriter);
  Value filter = inputOperands[1];
  Value packedFilter =
      (isConv2DNhwcHwcfOp)
          ? toPackLayoutRSCK_KCRSck(loc, filter, tiles, rewriter)
          : toPackLayoutKCRS_KCRSck(loc, filter, tiles, rewriter);
  SmallVector<Value, 2> packedInputs = {packedImage, packedFilter};

  // pack the output.
  Value output = outputOperands[0];
  Value packedOutput =
      (isConv2DNhwcHwcfOp)
          ? toPackLayoutNPQK_NKPQk(loc, output, tiles[0], rewriter)
          : toPackLayoutNCHW_NCHWc(loc, output, tiles[0], rewriter);

  SmallVector<int64_t, 2> strides = {1, 1};
  if (DenseIntElementsAttr stridesAttr = convOp.getStrides()) {
    auto strideValues = stridesAttr.getValues<int64_t>();
    assert(strideValues.size() == 2 && "expect two stride values");
    strides[0] = strideValues[0];
    strides[1] = strideValues[1];
  }

  // Swap convolution with generic.
  //         N   K   P   Q   k   C   R   S   c
  AffineExpr p1, p2, p3, p4, p5, r1, r2, r3, r4;
  bindDims(ctx, p1, p2, p3, p4, p5, r1, r2, r3, r4);
  AffineMap mapOut =
      AffineMap::get(/*dims=*/9, /*symbols=*/0, {p1, p2, p3, p4, p5}, ctx);
  AffineMap mapImg = AffineMap::get(
      /*dims=*/9, /*symbols=*/0,
      {p1, r1, p3 * strides[0] + r2, p4 * strides[1] + r3, r4}, ctx);
  AffineMap mapFil =
      AffineMap::get(/*dims=*/9, /*symbols=*/0, {p2, r1, r2, r3, r4, p5}, ctx);
  linalg::GenericOp replacementOp = rewriter.create<linalg::GenericOp>(
      loc, packedOutput.getType(), packedInputs, ValueRange{packedOutput},
      ArrayRef<AffineMap>{mapImg, mapFil, mapOut},
      ArrayRef<StringRef>{
          getParallelIteratorTypeName(), getParallelIteratorTypeName(),
          getParallelIteratorTypeName(), getParallelIteratorTypeName(),
          getParallelIteratorTypeName(), getReductionIteratorTypeName(),
          getReductionIteratorTypeName(), getReductionIteratorTypeName(),
          getReductionIteratorTypeName()},
      /*doc=*/"", /*libraryCall=*/"");
  rewriter.inlineRegionBefore(convOp->getRegion(0), replacementOp.getRegion(),
                              replacementOp.getRegion().begin());

  // convert back from pack layout.
  Value outPackedTensor = replacementOp.getResult(0);
  Value outUnPackedTensor = outputOperands[0];
  Value outReplacement =
      (isConv2DNhwcHwcfOp)
          ? fromPackLayoutNKPQk_NPQK(loc, outPackedTensor, outUnPackedTensor,
                                     tiles[0], rewriter)
          : fromPackLayoutNCHWc_NCHW(loc, outPackedTensor, outUnPackedTensor,
                                     tiles[0], rewriter);
  rewriter.replaceOp(convOp, outReplacement);
  return replacementOp;
}

//===----------------------------------------------------------------------===//
// Conv2DNhwcHwcfOp
//===----------------------------------------------------------------------===//
// Original layout: [N][P][Q][K] += [N][H][W][C] * [R][S][C][K]
// New      layout: [N][K'][P][Q][k] += [N][C'][H][W][c] * [K'][C'][R][S][c][k]
FailureOr<linalg::GenericOp>
mlir::linalgx::packConv2DNhwcHwcfOp(RewriterBase &rewriter,
                                    linalg::Conv2DNhwcHwcfOp convOp,
                                    ArrayRef<OpFoldResult> tiles) {
  return packConvolutions(rewriter, convOp, tiles);
}

//===----------------------------------------------------------------------===//
// Conv2DNchwFchwOp
//===----------------------------------------------------------------------===//
// Original layout: [N][K][P][Q] += [N][C][H][W] * [K][C][R][S]
// New      layout: [N][K'][P][Q][k] += [N][C'][H][W][c] + [K'][C'][R][S][c][k]
FailureOr<linalg::GenericOp>
mlir::linalgx::packConv2DNchwFchwOp(RewriterBase &rewriter,
                                    linalg::Conv2DNchwFchwOp convOp,
                                    ArrayRef<OpFoldResult> tiles) {
  return packConvolutions(rewriter, convOp, tiles);
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//
//  i      j        i     k      k      j
// [128 x 256] += [128 x 256] * [256 x 256]
//
// tile factor on i = 32
// tile factor on j = 16
// tile factor on k = 8
//
// [IB][JB][ib][jb] += [IB][KB][ib][kb] * [JB][KB][kb][jb]
// [4 ][16][32][16] += [4 ][32][32][8 ] * [16][32][8 ][16]
// KB is the batch reduce dimension.
FailureOr<linalg::GenericOp>
mlir::linalgx::packMatmulOp(RewriterBase &rewriter, linalg::MatmulOp matmulOp,
                            ArrayRef<OpFoldResult> tiles) {
  if (tiles.size() != 3)
    return rewriter.notifyMatchFailure(matmulOp, "require 3 tile factors");

  if (matmulOp.hasDynamicShape())
    return rewriter.notifyMatchFailure(matmulOp, "require static shape");

  if (matmulOp.hasBufferSemantics())
    return rewriter.notifyMatchFailure(matmulOp, "require tensor semantics");

  OpFoldResult tileOnI = tiles[0];
  OpFoldResult tileOnJ = tiles[1];
  OpFoldResult tileOnK = tiles[2];
  SmallVector<OpFoldResult, 2> tilesOnA = {tileOnI, tileOnK};
  SmallVector<OpFoldResult, 2> tilesOnB = {tileOnK, tileOnJ};
  SmallVector<OpFoldResult, 2> tilesOnC = {tileOnI, tileOnJ};

  Location loc = matmulOp.getLoc();
  // reshape input A and B.
  Value packedMatrixA =
      toPackLayoutNC_NCnc(loc, matmulOp.getInputs()[0], tilesOnA, rewriter);
  Value packedMatrixB =
      toPackLayoutKC_CKkc(loc, matmulOp.getInputs()[1], tilesOnB, rewriter);
  SmallVector<Value> packedInputs = {packedMatrixA, packedMatrixB};

  // reshape output C.
  Value packMatrixC =
      toPackLayoutNC_NCnc(loc, matmulOp.getOutputs()[0], tilesOnC, rewriter);

  // swap linalg.matmul with a linalg.generic.
  MLIRContext *ctx = matmulOp.getContext();
  AffineExpr p1, p2, r1, p3, p4, r2;
  bindDims(ctx, p1, p2, r1, p3, p4, r2);
  AffineMap mapA =
      AffineMap::get(/*dims=*/6, /*symbols=*/0, {p1, r1, p3, r2}, ctx);
  AffineMap mapB =
      AffineMap::get(/*dims=*/6, /*symbols=*/0, {p2, r1, r2, p4}, ctx);
  AffineMap mapC =
      AffineMap::get(/*dims=*/6, /*symbols=*/0, {p1, p2, p3, p4}, ctx);
  linalg::GenericOp replacementOp = rewriter.create<linalg::GenericOp>(
      loc, packMatrixC.getType(), packedInputs, ValueRange{packMatrixC},
      ArrayRef<AffineMap>{mapA, mapB, mapC},
      ArrayRef<StringRef>{
          getParallelIteratorTypeName(), getParallelIteratorTypeName(),
          getReductionIteratorTypeName(), getParallelIteratorTypeName(),
          getParallelIteratorTypeName(), getReductionIteratorTypeName()},
      /*doc=*/"", /*libraryCall=*/"");
  rewriter.inlineRegionBefore(matmulOp.getRegion(), replacementOp.getRegion(),
                              replacementOp.getRegion().begin());

  // convert back from pack layout.
  Value outPackTensor = replacementOp.getResult(0);
  Value outUnPackTensor = matmulOp.getOutputs()[0];
  Value outReplacement = fromPackLayoutNCnc_NC(
      loc, outPackTensor, outUnPackTensor, tilesOnC, rewriter);
  rewriter.replaceOp(matmulOp, outReplacement);
  return replacementOp;
}

namespace {

// Pack MatmulOp.
struct DoItOnMatmul : public OpRewritePattern<linalg::MatmulOp> {
  DoItOnMatmul(MLIRContext *context, ArrayRef<int64_t> blockingFactors,
               PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::MatmulOp>(context, benefit),
        blockingFactors(blockingFactors) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    FailureOr<linalg::GenericOp> packedMatmul = mlir::linalgx::packMatmulOp(
        rewriter, matmulOp,
        getAsOpFoldResult(rewriter.getI64ArrayAttr(blockingFactors)));
    if (failed(packedMatmul))
      return failure();
    return success();
  }

private:
  ArrayRef<int64_t> blockingFactors;
};

// From linalg.generic to linalg.matmul.
struct DeGeneralizeMatmul : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!linalgOp.hasTensorSemantics())
      return failure();
    if (!tpp::isMarkedWithTpp(linalgOp, "tpp.matmul"))
      return failure();
    SmallVector<Value> inputOperands = linalgOp.getDpsInputOperands();
    SmallVector<Value> outputOperands = linalgOp.getDpsInitOperands();
    rewriter.replaceOpWithNewOp<linalg::MatmulOp>(
        linalgOp, linalgOp.getResultTypes(), inputOperands, outputOperands);
    return success();
  }
};

// Entry point for packing a matmul operation.
// Pack MatmulOp as following:
// [NB][KB][nb][kb] += [NB][CB][nb][cb] * [KB][CB][cb][kb]
// CB = batch reduce dimension.
struct PackMatmul : public PackMatmulBase<PackMatmul> {
  PackMatmul() = default;
  PackMatmul(ArrayRef<int64_t> blockingFactors) {
    this->blockingFactors = blockingFactors;
  }

  void runOnOperation() override {
    if (blockingFactors.empty())
      return;
    MLIRContext *ctx = getOperation().getContext();
    RewritePatternSet patterns(ctx);
    mlir::tpp::populateSinkPackPatterns(patterns);
    patterns.add<DoItOnMatmul>(ctx, blockingFactors);
    patterns.add<DeGeneralizeMatmul>(ctx);
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
    FailureOr<linalg::GenericOp> genericOp =
        mlir::linalgx::packConv2DNchwFchwOp(
            rewriter, linalgOp,
            getAsOpFoldResult(rewriter.getI64ArrayAttr(blockingFactors)));
    if (failed(genericOp))
      return failure();
    return success();
  }

private:
  SmallVector<int64_t> blockingFactors;
};

struct PackConv2DNchwFchw : public PackConv2DNchwFchwBase<PackConv2DNchwFchw> {
  PackConv2DNchwFchw() = default;
  PackConv2DNchwFchw(ArrayRef<int64_t> blockingFactors) {
    this->blockingFactors = blockingFactors;
  }

  void runOnOperation() override {
    if (blockingFactors.empty())
      return;
    MLIRContext *ctx = getOperation().getContext();
    RewritePatternSet patterns(ctx);
    mlir::tpp::populateSinkPackPatterns(patterns);
    patterns.add<DoItOnConv2DNchwFchw>(ctx, blockingFactors);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

struct DoItOnConv2DNhwcHwcf
    : public OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
  DoItOnConv2DNhwcHwcf(MLIRContext *context, ArrayRef<int64_t> blockingFactors,
                       PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::Conv2DNhwcHwcfOp>(context, benefit),
        blockingFactors(blockingFactors) {}

  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp linalgOp,
                                PatternRewriter &rewriter) const override {
    FailureOr<linalg::GenericOp> maybeGeneric =
        mlir::linalgx::packConv2DNhwcHwcfOp(
            rewriter, linalgOp,
            getAsOpFoldResult(rewriter.getI64ArrayAttr(blockingFactors)));
    if (failed(maybeGeneric))
      return failure();
    return success();
  }

private:
  SmallVector<int64_t> blockingFactors;
};

struct PackConv2DNhwcHwcf : PackConv2DNhwcHwcfBase<PackConv2DNhwcHwcf> {
  PackConv2DNhwcHwcf() = default;
  PackConv2DNhwcHwcf(ArrayRef<int64_t> blockingFactors) {
    this->blockingFactors = blockingFactors;
  }

  void runOnOperation() override {
    if (blockingFactors.empty())
      return;
    MLIRContext *ctx = getOperation().getContext();
    RewritePatternSet patterns(ctx);
    mlir::tpp::populateSinkPackPatterns(patterns);
    patterns.add<DoItOnConv2DNhwcHwcf>(ctx, blockingFactors);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

//===----------------------------------------------------------------------===//
// PropagateThroughPadOp
//===----------------------------------------------------------------------===//

// Returns a vector that interchanges `elements` starting at offset `offset`
// based on the indexes in `interchangeVector`.
template <typename T>
SmallVector<T> interchange(ArrayRef<T> elements,
                           ArrayRef<int64_t> interchangeVector,
                           int offset = 0) {
  SmallVector<T> vec = llvm::to_vector(elements);
  for (auto en : llvm::enumerate(interchangeVector)) {
    vec[en.index() + offset] = elements[en.value() + offset];
  }
  return vec;
}

// The idea is to add as many zero padding dimensions in `high` and `low` based
// on the number of point loops.
struct PropagateThroughPadOp : public OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern<tensor::PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    Value inputPad = padOp.getSource();
    linalgx::UnPackOp unpackOp = inputPad.getDefiningOp<linalgx::UnPackOp>();
    if (!unpackOp)
      return failure();

    // no outer dim allowed.
    SmallVector<int64_t> outerDimsPerm =
        extractFromI64ArrayAttr(unpackOp.getOuterDimsPerm());
    if (!outerDimsPerm.empty())
      return failure();

    // bail out if one of the padded dimension is a tiled one.
    llvm::SmallBitVector paddedDims = padOp.getPaddedDims();
    SmallVector<int64_t> innerDimsPos =
        extractFromI64ArrayAttr(unpackOp.getInnerDimsPos());
    llvm::SmallBitVector innerDims(paddedDims.size());
    for (int64_t dim : innerDimsPos)
      paddedDims.flip(dim);
    if (paddedDims.anyCommon(innerDims))
      return failure();

    SmallVector<OpFoldResult> lowPad = padOp.getMixedLowPad();
    SmallVector<OpFoldResult> highPad = padOp.getMixedHighPad();
    size_t innerDimsPosSize = innerDimsPos.size();
    lowPad.append(innerDimsPosSize, rewriter.getIndexAttr(0));
    highPad.append(innerDimsPosSize, rewriter.getIndexAttr(0));

    auto newPadOp = rewriter.create<tensor::PadOp>(
        padOp.getLoc(), /*result type=*/nullptr, unpackOp.getInput(), lowPad,
        highPad, padOp.getNofold());
    SmallVector<Type> padArgsType(lowPad.size(), rewriter.getIndexType());
    SmallVector<Location> locs(lowPad.size(), padOp.getLoc());
    // Well, why this is not done by the builder?
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.createBlock(&newPadOp.getRegion(), newPadOp.getRegion().begin(),
                           padArgsType, locs);
      rewriter.create<tensor::YieldOp>(padOp.getLoc(),
                                       padOp.getConstantPaddingValue());
    }
    Value padOpRes = newPadOp.getResult();
    ShapedType padResultType = padOp.getResultType();
    Value outputUnPack = rewriter.create<tensor::EmptyOp>(
        padOp.getLoc(), padResultType.getShape(),
        padResultType.getElementType());
    Value replacement = toUnPackLayoutImpl(
        padOp.getLoc(), padOpRes, outputUnPack, unpackOp.getMixedTiles(),
        innerDimsPos, /*outer_dims_perm=*/{}, rewriter);

    rewriter.replaceOp(padOp, replacement);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PropagateThroughElementWiseOp
//===----------------------------------------------------------------------===//

// Propagate packing through element-wise linalg generic operation.
struct PropagateThroughElementWiseOp
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  // Further restrict to identity or minor identity maps.
  bool hasMinorIdentityMaps(linalg::GenericOp linalgOp) const {
    return llvm::all_of(linalgOp.getIndexingMapsArray(),
                        [](AffineMap map) { return map.isMinorIdentity(); });
  }

  // Require operands to come from a single `unpack` operation.
  bool hasOnlyOnePackedOperand(linalg::GenericOp linalgOp) const {
    unsigned count = 0;
    for (OpOperand &operand : linalgOp->getOpOperands()) {
      linalgx::UnPackOp unpackOp =
          operand.get().getDefiningOp<linalgx::UnPackOp>();
      if (unpackOp)
        count++;
    }
    return count == 1;
  }

  Value getPackOperand(OpOperand *operand, linalg::GenericOp linalgOp,
                       const DenseMap<int64_t, OpFoldResult> &dimAndTileMapping,
                       ArrayRef<int64_t> tileLoopPerm,
                       SmallVector<SmallVector<int64_t>> &outerPerms,
                       PatternRewriter &rewriter) const {
    linalgx::UnPackOp unpackOp =
        operand->get().getDefiningOp<linalgx::UnPackOp>();
    SmallVector<OpFoldResult> tiles;
    SmallVector<int64_t> innerDimsPos;
    SmallVector<int64_t> outerDimsPerm;
    // If the operand comes from an unpack operation simply pack the operand
    // with the same tiles, and dimsPos extracted from the unpack, otherwise
    // infer them from `dimsAndTileMapping`.
    if (unpackOp) {
      tiles = unpackOp.getMixedTiles();
      innerDimsPos = extractFromI64ArrayAttr(unpackOp.getInnerDimsPos());
      outerDimsPerm = extractFromI64ArrayAttr(unpackOp.getOuterDimsPerm());
    } else {
      AffineMap mapOperand = linalgOp.getMatchingIndexingMap(operand);
      for (unsigned pos = 0; pos < mapOperand.getNumResults(); pos++) {
        unsigned posInDomain = mapOperand.getDimPosition(pos);
        if (dimAndTileMapping.count(posInDomain)) {
          tiles.push_back(dimAndTileMapping.lookup(posInDomain));
          innerDimsPos.push_back(pos);
        }
      }
      // Handle `outer_dims_perm`. See example:
      // current map      : (d0, d1, d2, d3) -> (d2, d3)
      // dimAndTileMapping: dim | tile
      //                    3   | 32
      // tileLoopPerm     : [0, 3, 1, 2]
      // First map d2, d3 with their position in the array as:
      // currentPositionTileLoops: dim | pos
      //                           d2  | 0
      //                           d3  | 1
      // then scan `tileLoopPerm` in order and get the `outer_dims_perm`
      // to be used, here it would be [1, 0].
      DenseMap<int64_t, int64_t> currentPositionTileLoops;
      for (unsigned pos = 0; pos < mapOperand.getNumResults(); pos++) {
        unsigned posInDomain = mapOperand.getDimPosition(pos);
        currentPositionTileLoops[posInDomain] = pos;
      }
      for (int64_t loopIdx : tileLoopPerm) {
        if (currentPositionTileLoops.count(loopIdx))
          outerDimsPerm.push_back(currentPositionTileLoops.lookup(loopIdx));
      }
    }
    // save the outer perm for later, when we compute the map.
    outerPerms.push_back(outerDimsPerm);
    return toPackLayoutImpl(linalgOp.getLoc(), operand->get(), tiles,
                            innerDimsPos, outerDimsPerm, rewriter);
  }

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {

    if (!linalg::isElementwise(linalgOp))
      return rewriter.notifyMatchFailure(linalgOp,
                                         "expects an elementwise operation");

    if (!hasMinorIdentityMaps(linalgOp))
      return rewriter.notifyMatchFailure(
          linalgOp, "expects all identity/minor identity maps");

    // Require only one operand to come from an `unpack` operation.
    if (!hasOnlyOnePackedOperand(linalgOp))
      return rewriter.notifyMatchFailure(linalgOp,
                                         "expects a single packed operand");

    // Pack and unpack operate on result of each operand map in the linalg
    // operation. We need to map these dimensions (co-domain) to the domain of
    // the linalg operation. Scan each input and output operands. For each map
    // associated to the operand check the equivalent dimension in the domain
    // and bind it with the tile size.
    DenseMap<int64_t, OpFoldResult> dimAndTileMapping;
    SmallVector<int64_t> tileLoopPerms;
    SmallVector<int64_t> interchangeVector;
    for (OpOperand &operand : linalgOp->getOpOperands()) {
      linalgx::UnPackOp unpackOp =
          operand.get().getDefiningOp<linalgx::UnPackOp>();
      if (!unpackOp)
        continue;

      // Handle outer dims perm. If the given tensor dimension is permuted get
      // the domain dimension and use it to infer which co-domain dimension is
      // permuted in `getPackedOperand`.
      SmallVector<int64_t> outerDimsPerm =
          extractFromI64ArrayAttr(unpackOp.getOuterDimsPerm());
      if (!outerDimsPerm.empty()) {
        tileLoopPerms = outerDimsPerm;
        interchangeVector = outerDimsPerm;
        // tileLoopPerms represents the new permutation of the tiled loops and
        // need to cover all the current loop index range, and must be a valid
        // permutation.
        if (tileLoopPerms.size() != linalgOp.getNumLoops())
          return failure();
        if (!linalg::isPermutation(tileLoopPerms))
          return failure();
      }

      SmallVector<int64_t> innerDimsPos =
          extractFromI64ArrayAttr(unpackOp.getInnerDimsPos());
      assert(innerDimsPos.size() && "expect non-empty");
      for (int64_t dimPos : innerDimsPos) {
        interchangeVector.push_back(dimPos + innerDimsPos.size());
      }

      // map *domain* of linalg operation to tiles.
      DenseMap<int64_t, OpFoldResult> currentDimAndTileMapping =
          unpackOp.getDimAndTileMapping();
      AffineMap mapOperand = linalgOp.getMatchingIndexingMap(&operand);
      for (unsigned posInCodomain = 0;
           posInCodomain < mapOperand.getNumResults(); posInCodomain++) {
        // fail if we dealing with 'complex' affine maps. Only dim expression
        // are accepted.
        if (!mapOperand.getResult(posInCodomain).isa<AffineDimExpr>())
          return failure();
        unsigned posInDomain = mapOperand.getDimPosition(posInCodomain);
        if (currentDimAndTileMapping.count(posInCodomain))
          dimAndTileMapping[posInDomain] =
              currentDimAndTileMapping[posInCodomain];
      }
    }

    // no work to do, exit. We did not find any unpacked input or output
    // operands.
    if (dimAndTileMapping.empty()) {
      return failure();
    }

    // Bail out if `interchangeVector` is not a valid permutation. We need to
    // transpose only if `tileLoopPerms` is not empty.
    if (!tileLoopPerms.empty() && !linalg::isPermutation(interchangeVector)) {
      return failure();
    }

    SmallVector<SmallVector<int64_t>> outerPermsForMaps;
    SmallVector<Value> packedInputOperands;
    for (OpOperand *operand : linalgOp.getDpsInputOperands()) {
      Value packedOperand =
          getPackOperand(operand, linalgOp, dimAndTileMapping, tileLoopPerms,
                         outerPermsForMaps, rewriter);
      packedInputOperands.push_back(packedOperand);
    }

    SmallVector<Value> packedOutputOperands;
    SmallVector<Type> packedOutputTypes;
    SmallVector<Value> unpackOutputs;
    for (OpOperand *operand : linalgOp.getDpsInitOperands()) {
      Value packedOperand =
          getPackOperand(operand, linalgOp, dimAndTileMapping, tileLoopPerms,
                         outerPermsForMaps, rewriter);
      packedOutputOperands.push_back(packedOperand);
      packedOutputTypes.push_back(packedOperand.getType());
      linalgx::UnPackOp unpackOp =
          operand->get().getDefiningOp<linalgx::UnPackOp>();
      if (unpackOp)
        unpackOutputs.push_back(unpackOp.getOutput());
    }

    // FIXME: innerDimsPos and outerDimsPerm must have been unsigned in the
    // first place. `AffineMap::getPermutationMap` requires unsigned, convert.
    AffineMap permutationMap;
    if (!tileLoopPerms.empty()) {
      SmallVector<unsigned> interchangeVectorUInt;
      for (int64_t i : interchangeVector)
        interchangeVectorUInt.push_back(
            static_cast<std::make_unsigned<unsigned>::type>(i));
      permutationMap = inversePermutation(AffineMap::getPermutationMap(
          interchangeVectorUInt, linalgOp.getContext()));
    }
    assert(outerPermsForMaps.size() == linalgOp->getNumOperands());

    unsigned packedDims = dimAndTileMapping.size();
    SmallVector<AffineMap> newMaps;
    // Get the new map for each operand.
    for (OpOperand &operand : linalgOp->getOpOperands()) {
      AffineMap mapOperand = linalgOp.getMatchingIndexingMap(&operand);
      unsigned numSymbols = 0;
      unsigned numDims = linalgOp.getNumLoops() + packedDims;
      unsigned oldResultExprs = mapOperand.getNumResults();
      SmallVector<AffineExpr> dimLoops(oldResultExprs,
                                       rewriter.getAffineDimExpr(-1));
      SmallVector<AffineExpr> dimPointLoops;
      unsigned pointLoopIdx = 0;
      for (unsigned posInCodomain = 0; posInCodomain < oldResultExprs;
           posInCodomain++) {
        unsigned posInDomain = mapOperand.getDimPosition(posInCodomain);
        if (dimAndTileMapping.count(posInDomain))
          dimPointLoops.push_back(rewriter.getAffineDimExpr(
              linalgOp.getNumLoops() + pointLoopIdx++));
        dimLoops[posInCodomain] = rewriter.getAffineDimExpr(posInDomain);
      }
      if (!tileLoopPerms.empty())
        dimLoops = interchange<AffineExpr>(
            dimLoops, outerPermsForMaps[operand.getOperandNumber()]);
      dimLoops.append(dimPointLoops);
      AffineMap newMap =
          AffineMap::get(numDims, numSymbols, dimLoops, linalgOp.getContext());
      // Apply the transposition on the iterators.
      if (!tileLoopPerms.empty()) {
        assert(permutationMap && "must be valid");
        newMap = newMap.compose(permutationMap);
      }
      newMaps.push_back(newMap);
    }

    SmallVector<StringRef> newIteratorTypes(linalgOp.getNumLoops() + packedDims,
                                            getParallelIteratorTypeName());

    linalg::GenericOp replacementOp = rewriter.create<linalg::GenericOp>(
        linalgOp.getLoc(), packedOutputTypes, packedInputOperands,
        packedOutputOperands, newMaps, newIteratorTypes, /*docs=*/"",
        /*libraryCall=*/"");
    rewriter.inlineRegionBefore(linalgOp.getRegion(), replacementOp.getRegion(),
                                replacementOp.getRegion().begin());

    SmallVector<Value> outReplacements;
    size_t idx = 0;
    for (OpOperand *operand : replacementOp.getDpsInitOperands()) {
      linalgx::PackOp packOp = operand->get().getDefiningOp<linalgx::PackOp>();
      Value result = replacementOp.getTiedOpResult(operand);
      if (unpackOutputs.empty())
        outReplacements.push_back(toUnPackLayoutImpl(
            linalgOp.getLoc(), result, linalgOp.getOutputs()[idx++],
            packOp.getMixedTiles(),
            extractFromI64ArrayAttr(packOp.getInnerDimsPos()),
            extractFromI64ArrayAttr(packOp.getOuterDimsPerm()), rewriter));
      else
        outReplacements.push_back(toUnPackLayoutImpl(
            linalgOp.getLoc(), result, unpackOutputs[idx++],
            packOp.getMixedTiles(),
            extractFromI64ArrayAttr(packOp.getInnerDimsPos()),
            extractFromI64ArrayAttr(packOp.getOuterDimsPerm()), rewriter));
    }
    rewriter.replaceOp(linalgOp, outReplacements);
    return success();
  }
};

} // end namespace

void mlir::tpp::populateSinkPackPatterns(RewritePatternSet &patterns) {
  patterns.add<PropagateThroughElementWiseOp, PropagateThroughPadOp>(
      patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::tpp::createPackMatmulPass() {
  return std::make_unique<PackMatmul>();
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createPackConv2DNchwFchwPass() {
  return std::make_unique<PackConv2DNchwFchw>();
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createPackConv2DNhwcHwcfPass() {
  return std::make_unique<PackConv2DNhwcHwcf>();
}
