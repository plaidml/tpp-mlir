//===- ConvertToBlockLayoutAndBack.cpp ---------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

// Helper function to create the pack operation.
static Value toPackLayoutImpl(OpBuilder &builder, Location loc, Value input,
                              ArrayRef<OpFoldResult> tiles,
                              ArrayRef<int64_t> innerDimsPos,
                              ArrayRef<int64_t> outerDimsPerm) {
  SmallVector<Value> dynamicTiles;
  SmallVector<int64_t> staticTiles;
  dispatchIndexOpFoldResults(tiles, dynamicTiles, staticTiles);
  RankedTensorType result =
      tensor::PackOp::inferPackedType(input.getType().cast<RankedTensorType>(),
                                      staticTiles, innerDimsPos, outerDimsPerm);
  auto inputType = input.getType().cast<RankedTensorType>();
  ArrayRef<int64_t> shape = result.getShape();
  Value output =
      builder.create<tensor::EmptyOp>(loc, shape, inputType.getElementType());
  return builder.create<tensor::PackOp>(loc, input, output, innerDimsPos, tiles,
                                        /*paddingValue=*/std::nullopt,
                                        outerDimsPerm);
}

// Helper function to create the unpack operation.
static Value toUnPackLayoutImpl(OpBuilder &builder, Location loc, Value input,
                                Value output, ArrayRef<OpFoldResult> tiles,
                                ArrayRef<int64_t> innerDimPos,
                                ArrayRef<int64_t> outerDimsPerm) {
  if (auto fillOp = output.getDefiningOp<linalg::FillOp>())
    output = fillOp.getOutputs()[0];
  return builder.create<tensor::UnPackOp>(loc, input, output, innerDimPos,
                                          tiles, outerDimsPerm);
}

static Value handleLayout_VNNI(OpBuilder &builder, Location loc, Value input,
                               ArrayRef<OpFoldResult> tiles) {
  assert(tiles.size() == 1 && "expect 1 block for VNNI");
  SmallVector<int64_t> innerDimPos = {
      input.getType().cast<ShapedType>().getRank() - 2};
  return toPackLayoutImpl(builder, loc, input, tiles, innerDimPos,
                          /*outerDimsPerm=*/{});
}

static Value handleBRGemmLayout_VNNI(OpBuilder &builder, Location loc,
                                     Value input,
                                     ArrayRef<OpFoldResult> tiles) {
  assert(tiles.size() == 1 && "expect 1 block for VNNI");
  SmallVector<int64_t> innerDimPos = {1};
  return toPackLayoutImpl(builder, loc, input, tiles, innerDimPos,
                          /*outerDimsPerm=*/{});
}

// Helper function to pack from NC to [N/2][C][2].
static Value toPackLayout_VNNI(OpBuilder &builder, Location loc, Value input,
                               ArrayRef<OpFoldResult> tiles) {
  return handleLayout_VNNI(builder, loc, input, tiles);
}

// Helper function to pack from [N][K][C] to [N][K/2][C][2].
static Value toPackBRGemmLayout_VNNI(OpBuilder &builder, Location loc,
                                     Value input,
                                     ArrayRef<OpFoldResult> tiles) {
  return handleBRGemmLayout_VNNI(builder, loc, input, tiles);
}

static Value handleLayoutNCHW_NCHWc(OpBuilder &builder, Location loc,
                                    Value input, Value output,
                                    ArrayRef<OpFoldResult> tiles) {
  assert(tiles.size() == 1 && "expect one tile size for NCHW_NCHWc");
  SmallVector<int64_t> innerDimPos = {1};
  if (!output)
    return toPackLayoutImpl(builder, loc, input, tiles, innerDimPos,
                            /*outerDimsPerm=*/{});
  return toUnPackLayoutImpl(builder, loc, input, output, tiles, innerDimPos,
                            /*outerDimsPerm=*/{});
}

// Helper function to pack from NCHW to NCHWc.
static Value toPackLayoutNCHW_NCHWc(OpBuilder &builder, Location loc,
                                    Value input, ArrayRef<OpFoldResult> tiles) {
  return handleLayoutNCHW_NCHWc(builder, loc, input, nullptr, tiles);
}

// Helper function to unpack from NCHWc to NCHW.
static Value fromPackLayoutNCHWc_NCHW(OpBuilder &builder, Location loc,
                                      Value input, Value output,
                                      ArrayRef<OpFoldResult> tiles) {
  return handleLayoutNCHW_NCHWc(builder, loc, input, output, tiles);
}

static Value handleLayoutNPQK_NKPQk(OpBuilder &builder, Location loc,
                                    Value input, Value output,
                                    ArrayRef<OpFoldResult> tiles) {
  assert(tiles.size() == 1 && "expect one tile size for NPQK_NKPQk");
  SmallVector<int64_t> innerDimsPos = {3};
  SmallVector<int64_t> outerDimsPerm = {0, 3, 1, 2};
  if (!output)
    return toPackLayoutImpl(builder, loc, input, tiles, innerDimsPos,
                            outerDimsPerm);
  return toUnPackLayoutImpl(builder, loc, input, output, tiles, innerDimsPos,
                            outerDimsPerm);
}

// Helper function to pack NPQK to NKPQk.
static Value toPackLayoutNPQK_NKPQk(OpBuilder &builder, Location loc,
                                    Value input, ArrayRef<OpFoldResult> tiles) {
  return handleLayoutNPQK_NKPQk(builder, loc, input, nullptr, tiles);
}

// Helper function to unpack NKPQk to NPQK.
static Value fromPackLayoutNKPQk_NPQK(OpBuilder &builder, Location loc,
                                      Value input, Value output,
                                      ArrayRef<OpFoldResult> tiles) {
  return handleLayoutNPQK_NKPQk(builder, loc, input, output, tiles);
}

// Helper function to pack from RSCK to KCRSck.
static Value toPackLayoutRSCK_KCRSck(OpBuilder &builder, Location loc,
                                     Value input,
                                     ArrayRef<OpFoldResult> tiles) {
  assert(tiles.size() == 2 && "expect two tiles for RSCK_KCRSck");
  SmallVector<int64_t> innerDimsPos = {2, 3};
  SmallVector<int64_t> outerDimsPerm = {3, 2, 0, 1};
  return toPackLayoutImpl(builder, loc, input, tiles, innerDimsPos,
                          outerDimsPerm);
}

// Helper function to pack from KCRS to KCRSck.
static Value toPackLayoutKCRS_KCRSck(OpBuilder &builder, Location loc,
                                     Value input,
                                     ArrayRef<OpFoldResult> tiles) {
  assert(tiles.size() == 2 && "expect two tiles size for KCRS_KCRSck");
  SmallVector<int64_t> innerDimPos = {1, 0};
  return toPackLayoutImpl(builder, loc, input, tiles, innerDimPos,
                          /*outerDimsPerm=*/{});
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
      static_cast<bool>(std::is_same<OpTy, linalg::Conv2DNhwcHwcfOp>::value);

  Location loc = convOp.getLoc();
  MLIRContext *ctx = convOp.getContext();

  SmallVector<Value> inputOperands = convOp.getDpsInputs();
  SmallVector<Value> outputOperands = convOp.getDpsInits();

  // pack the image and the filter.
  Value image = inputOperands[0];
  Value packedImage =
      (isConv2DNhwcHwcfOp)
          ? toPackLayoutNPQK_NKPQk(rewriter, loc, image, tiles[0])
          : toPackLayoutNCHW_NCHWc(rewriter, loc, image, tiles[0]);
  Value filter = inputOperands[1];
  Value packedFilter =
      (isConv2DNhwcHwcfOp)
          ? toPackLayoutRSCK_KCRSck(rewriter, loc, filter, tiles)
          : toPackLayoutKCRS_KCRSck(rewriter, loc, filter, tiles);
  SmallVector<Value, 2> packedInputs = {packedImage, packedFilter};

  // pack the output.
  Value output = outputOperands[0];
  Value packedOutput =
      (isConv2DNhwcHwcfOp)
          ? toPackLayoutNPQK_NKPQk(rewriter, loc, output, tiles[0])
          : toPackLayoutNCHW_NCHWc(rewriter, loc, output, tiles[0]);

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
      ArrayRef<utils::IteratorType>{
          utils::IteratorType::parallel, utils::IteratorType::parallel,
          utils::IteratorType::parallel, utils::IteratorType::parallel,
          utils::IteratorType::parallel, utils::IteratorType::reduction,
          utils::IteratorType::reduction, utils::IteratorType::reduction,
          utils::IteratorType::reduction},
      /*doc=*/"", /*libraryCall=*/"");
  rewriter.inlineRegionBefore(convOp->getRegion(0), replacementOp.getRegion(),
                              replacementOp.getRegion().begin());
  if (auto metadata = convOp->getAttr("metadata"))
    replacementOp->setAttr("metadata", metadata);

  // convert back from pack layout.
  Value outPackedTensor = replacementOp.getResult(0);
  Value outUnPackedTensor = outputOperands[0];
  Value outReplacement =
      (isConv2DNhwcHwcfOp)
          ? fromPackLayoutNKPQk_NPQK(rewriter, loc, outPackedTensor,
                                     outUnPackedTensor, tiles[0])
          : fromPackLayoutNCHWc_NCHW(rewriter, loc, outPackedTensor,
                                     outUnPackedTensor, tiles[0]);
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
  if (!linalgx::utils::validateFullTilesOnDims(
          cast<TilingInterface>(convOp.getOperation()), tiles,
          {/*Kidx=*/3, /*Cidx=*/6}))
    return rewriter.notifyMatchFailure(convOp, "expect full tiles only");
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
  if (!linalgx::utils::validateFullTilesOnDims(
          cast<TilingInterface>(convOp.getOperation()), tiles,
          {/*Kidx=*/1, /*Cidx=*/4}))
    return rewriter.notifyMatchFailure(convOp, "expect full tiles only");
  return packConvolutions(rewriter, convOp, tiles);
}

template <typename OpTy>
static FailureOr<linalg::LinalgOp>
packMatmulOpImpl(RewriterBase &rewriter, OpTy matmulOp,
                 ArrayRef<OpFoldResult> tiles) {
  static_assert(
      llvm::is_one_of<OpTy, linalg::MatmulOp, linalg::BatchMatmulOp>::value,
      "applies to only matmul or batch matmul operations");

  OpBuilder::InsertionGuard guard(rewriter);
  // The op is replaced, we need to set the insertion
  // point after it.
  rewriter.setInsertionPointAfter(matmulOp);

  if (matmulOp.hasDynamicShape())
    return rewriter.notifyMatchFailure(matmulOp, "require static shape");

  if (matmulOp.hasBufferSemantics())
    return rewriter.notifyMatchFailure(matmulOp, "require tensor semantics");

  OpFoldResult tileOnI = tiles[0];
  OpFoldResult tileOnJ = tiles[1];
  OpFoldResult tileOnK = tiles[2];
  bool isBatchMatmulOp = std::is_same_v<OpTy, linalg::BatchMatmulOp>;
  size_t inc = isBatchMatmulOp ? 1 : 0;
  size_t posI = 0 + inc;
  size_t posJ = 1 + inc;
  size_t posK = 2 + inc;
  if (!linalgx::utils::validateFullTilesOnDims(
          cast<TilingInterface>(matmulOp.getOperation()),
          {tileOnI, tileOnJ, tileOnK}, {posI, posJ, posK})) {
    return rewriter.notifyMatchFailure(matmulOp, "expect full tiles only");
  }

  // [..][IB][JB][ib][jb] += [..][IB][KB][ib][kb] * [..][KB][JB][jb][kb]
  auto packedCanonicalMatmul = linalg::packMatmulGreedily(
      rewriter, matmulOp, tiles, /*mnkPaddedSizesNextMultipleOf=*/{},
      /*mnkPackedSizes=*/{0, 1, 2});
  if (failed(packedCanonicalMatmul))
    return failure();

  assert(packedCanonicalMatmul->packOps.size() == 3);
  assert(packedCanonicalMatmul->unPackOps.size() == 1);

  SmallVector<int64_t> innerPerm = {1, 0};
  SmallVector<int64_t> outerPerm = {1, 0};
  if (isBatchMatmulOp)
    outerPerm = {0, 2, 1};
  auto packedMatmul =
      linalg::packTranspose(rewriter, packedCanonicalMatmul->packOps[1],
                            packedCanonicalMatmul->packedLinalgOp,
                            /*maybeUnPackOp=*/nullptr, outerPerm, innerPerm);
  if (failed(packedMatmul))
    return failure();
  return packedMatmul->transposedLinalgOp;
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
FailureOr<linalg::LinalgOp>
mlir::linalgx::packMatmulOp(RewriterBase &rewriter, linalg::MatmulOp matmulOp,
                            ArrayRef<OpFoldResult> tiles) {
  if (tiles.size() != 3)
    return rewriter.notifyMatchFailure(matmulOp, "require 3 tile factors");

  return packMatmulOpImpl<linalg::MatmulOp>(rewriter, matmulOp, tiles);
}

//===----------------------------------------------------------------------===//
// BatchMatmulOp
//===----------------------------------------------------------------------===//
// Original layout:
//  [B][I][J] += [B][I][K] * [B][K][J]
// New layout:
//  [B][IB][JB][ib][jb] += [B][IB][KB][ib][kb] * [B][JB][KB][kb][jb]
FailureOr<linalg::LinalgOp>
mlir::linalgx::packMatmulOp(RewriterBase &rewriter,
                            linalg::BatchMatmulOp matmulOp,
                            ArrayRef<OpFoldResult> tiles) {
  if (tiles.size() != 3)
    return rewriter.notifyMatchFailure(matmulOp, "require 3 tile factors");

  return packMatmulOpImpl<linalg::BatchMatmulOp>(rewriter, matmulOp, tiles);
}

//===----------------------------------------------------------------------===//
// MatmulOp (VNNI packing)
//===----------------------------------------------------------------------===//
// Original layout:
//      [IB][JB][ib][jb] += [IB][KB][ib][kb] * [JB][KB][kb][jb]
// New      layout:
//      [IB][JB][ib][jb] += [IB][KB][ib][kb] * [JB][KB][kb/VNNI][jb][VNNI]
FailureOr<linalg::GenericOp>
mlir::linalgx::packVNNIMatmulOp(RewriterBase &rewriter,
                                linalg::GenericOp matmulOp) {
  if (matmulOp.getInputs().size() > 0) {
    auto elementType = getElementTypeOrSelf(matmulOp.getInputs()[0].getType());
    if (!elementType.isBF16())
      return rewriter.notifyMatchFailure(matmulOp, "require bf16 type");
  }

  if (matmulOp.hasDynamicShape())
    return rewriter.notifyMatchFailure(matmulOp, "require static shape");

  if (matmulOp.hasBufferSemantics())
    return rewriter.notifyMatchFailure(matmulOp, "require tensor semantics");

  if (failed(linalgx::utils::isContraction(matmulOp)))
    return rewriter.notifyMatchFailure(matmulOp, "require matmul semantics");

  Value operandB = matmulOp.getInputs()[1];
  if (operandB.getType().cast<ShapedType>().getRank() != 4)
    return rewriter.notifyMatchFailure(matmulOp, "already packed to VNNI");

  Location loc = matmulOp.getLoc();
  auto blockingFactor = vnni::utils::getVnniBlockingFactor(operandB.getType());
  if (!blockingFactor) {
    return rewriter.notifyMatchFailure(matmulOp,
                                       "unsupported blocking factor for type");
  }
  SmallVector<OpFoldResult, 1> tilesOnSmallK = {
      rewriter.getI64IntegerAttr(*blockingFactor)};
  // reshape input B.
  Value packedMatrixB =
      toPackLayout_VNNI(rewriter, loc, operandB, tilesOnSmallK);
  MLIRContext *ctx = matmulOp.getContext();
  AffineExpr p1, p2, r1, p3, p4, r2, r3;
  SmallVector<Value> packedInputs = {matmulOp.getInputs()[0], packedMatrixB};
  AffineMap mapA, mapB, mapC;
  Value matrixC = matmulOp.getOutputs()[0];

  bindDims(ctx, p1, p2, r1, r3, p3, p4, r2);
  mapA = AffineMap::get(/*dims=*/7, /*symbols=*/0, {p1, r1, p3, r2}, ctx);
  mapB = AffineMap::get(/*dims=*/7, /*symbols=*/0,
                        {p2, r1, r2.floorDiv(*blockingFactor), p4, r3}, ctx);
  mapC = AffineMap::get(/*dims=*/7, /*symbols=*/0, {p1, p2, p3, p4}, ctx);
  auto replacementOp = rewriter.create<linalg::GenericOp>(
      loc, matrixC.getType(), packedInputs, ValueRange{matrixC},
      ArrayRef<AffineMap>{mapA, mapB, mapC},
      ArrayRef<mlir::utils::IteratorType>{mlir::utils::IteratorType::parallel,
                                          mlir::utils::IteratorType::parallel,
                                          mlir::utils::IteratorType::reduction,
                                          mlir::utils::IteratorType::reduction,
                                          mlir::utils::IteratorType::parallel,
                                          mlir::utils::IteratorType::parallel,
                                          mlir::utils::IteratorType::reduction},
      /*doc=*/"", /*libraryCall=*/"");

  rewriter.inlineRegionBefore(matmulOp.getRegion(), replacementOp.getRegion(),
                              replacementOp.getRegion().begin());

  rewriter.replaceOp(matmulOp, replacementOp.getResult(0));
  return replacementOp;
}

//===----------------------------------------------------------------------===//
// BrgemmOp (VNNI layout)
//===----------------------------------------------------------------------===//
// Original layout: [I][J] += [R][I][K] * [R][K][J]
// New      layout: [I][J] += [R][I][K] * [R][K/VNNI][J][VNNI]
FailureOr<tpp::BrgemmOp>
mlir::linalgx::packVNNIBRGemmOp(RewriterBase &rewriter,
                                linalg::BatchReduceMatmulOp brgemmOp) {
  auto elementType = getElementTypeOrSelf(brgemmOp.getInputs()[0].getType());
  if (!elementType.isBF16())
    return rewriter.notifyMatchFailure(brgemmOp, "require bf16 type");

  if (brgemmOp.hasDynamicShape())
    return rewriter.notifyMatchFailure(brgemmOp, "require static shape");

  if (brgemmOp.hasBufferSemantics())
    return rewriter.notifyMatchFailure(brgemmOp, "require tensor semantics");

  Value operandB = brgemmOp.getInputs()[1];
  // Blocking factor on the `k` dimension.
  auto blockingFactor = vnni::utils::getVnniBlockingFactor(operandB.getType());
  if (!blockingFactor) {
    return rewriter.notifyMatchFailure(brgemmOp,
                                       "unsupported blocking factor for type");
  }
  SmallVector<OpFoldResult, 1> tilesOnK = {rewriter.getI64IntegerAttr(2)};

  Location loc = brgemmOp.getLoc();
  // Reshape input B.
  Value packedMatrixB =
      toPackBRGemmLayout_VNNI(rewriter, loc, operandB, tilesOnK);
  auto replacementOp = rewriter.create<tpp::BrgemmOp>(
      loc,
      ValueRange{brgemmOp.getInputs()[0], packedMatrixB,
                 brgemmOp.getOutputs()[0]},
      brgemmOp.getOutputs()[0].getType());
  rewriter.replaceOp(brgemmOp, replacementOp.getResult(0));
  return replacementOp;
}

namespace {

//===----------------------------------------------------------------------===//
// BubbleUpThroughFillOp
//===----------------------------------------------------------------------===//

// Attempt to avoid packing a fill op. Instead create a 'packed' fill.
// %0 = tensor.empty
// %packed = tensor.empty
// %1 = linalg.fill ins(%cst) outs(%0)
// %2 = tensor.pack %1 into %packed
// %3 = some_packed_op %2
//
// --->
//
// %0 = tensor.empty
// %1 = linalg.fill ins(%cst) outs (%packed)
// %2 = some_packed_op %1
// %3 = tensor.unpack %2 into %0
//
struct BubbleUpThroughFillOp : public OpRewritePattern<tensor::PackOp> {
  using OpRewritePattern<tensor::PackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PackOp packOp,
                                PatternRewriter &rewriter) const override {
    Value source = packOp.getSource();
    auto fillOp = source.getDefiningOp<linalg::FillOp>();
    if (!fillOp)
      return failure();

    Value fillRes = fillOp.getResult(0);
    if (!fillRes.hasOneUse())
      return failure();

    // Replace result with output.
    rewriter.replaceAllUsesWith(fillRes, fillOp.getOutputs()[0]);
    auto empty = tensor::PackOp::createDestinationTensor(
        rewriter, packOp.getLoc(), source, packOp.getMixedTiles(),
        packOp.getInnerDimsPos(), packOp.getOuterDimsPerm());
    rewriter.replaceOpWithNewOp<linalg::FillOp>(packOp, fillOp.getInputs(),
                                                empty);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

// Pack MatmulOp and BatchMatmulOp.
template <typename OpTy> struct PackMatmulImpl : public OpRewritePattern<OpTy> {
  PackMatmulImpl(MLIRContext *context, ArrayRef<int64_t> blockingFactors,
                 PatternBenefit benefit = 1)
      : OpRewritePattern<OpTy>(context, benefit),
        blockingFactors(blockingFactors) {}

  LogicalResult matchAndRewrite(OpTy matmulOp,
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

// Entry point for packing a matmul operation.
// Pack MatmulOp as following:
// [NB][KB][nb][kb] += [NB][CB][nb][cb] * [KB][CB][cb][kb]
// CB = batch reduce dimension.
// Pack a BatchMatmulOp as following:
// [B][IB][JB][ib][jb] += [B][IB][KB][ib][kb] * [B][JB][KB][kb][jb]
// KB is the batch reduce dimension.
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
    patterns.add<PackMatmulImpl<linalg::MatmulOp>,
                 PackMatmulImpl<linalg::BatchMatmulOp>>(ctx, blockingFactors);
    linalg::populateLinalgDeGeneralizationPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
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
    patterns.add<DoItOnConv2DNchwFchw>(ctx, blockingFactors);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
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
    patterns.add<DoItOnConv2DNhwcHwcf>(ctx, blockingFactors);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

// Pack MatmulOp to VNNI.
struct VNNIOnMatmul : public OpRewritePattern<linalg::GenericOp> {
  VNNIOnMatmul(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::GenericOp>(context, benefit) {}
  LogicalResult matchAndRewrite(linalg::GenericOp matmulOp,
                                PatternRewriter &rewriter) const override {
    FailureOr<linalg::GenericOp> packedMatmul =
        mlir::linalgx::packVNNIMatmulOp(rewriter, matmulOp);
    if (failed(packedMatmul))
      return failure();
    return success();
  }
};

// Pack BRGemmOp to VNNI.
struct VNNIOnBRGemm : public OpRewritePattern<linalg::BatchReduceMatmulOp> {
  VNNIOnBRGemm(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::BatchReduceMatmulOp>(context, benefit) {}
  LogicalResult matchAndRewrite(linalg::BatchReduceMatmulOp brgemmOp,
                                PatternRewriter &rewriter) const override {
    FailureOr<tpp::BrgemmOp> packedBRGemm =
        mlir::linalgx::packVNNIBRGemmOp(rewriter, brgemmOp);
    if (failed(packedBRGemm))
      return failure();
    return success();
  }
};

// Entry point for packing a matmul/brgemm operation to vnni format.
struct PackVNNI : public PackVNNIBase<PackVNNI> {
  PackVNNI() = default;

  void runOnOperation() override {
    MLIRContext *ctx = getOperation().getContext();
    RewritePatternSet patterns(ctx);
    linalg::populateLinalgDeGeneralizationPatterns(patterns);
    patterns.add<VNNIOnMatmul, VNNIOnBRGemm>(ctx);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

struct PropagatePackUnPack
    : public PropagatePackUnPackBase<PropagatePackUnPack> {
  void runOnOperation() override {
    MLIRContext *ctx = getOperation().getContext();
    RewritePatternSet patterns(ctx);
    tpp::populateSinkPackPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

// TODO: (lorenzo) Upstream in `populateSimplifyTensorPack`.
// Do not pack an empty.
struct SimplifyPackToEmpty : public OpRewritePattern<tensor::PackOp> {
  using OpRewritePattern<tensor::PackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PackOp packOp,
                                PatternRewriter &rewriter) const override {
    auto emptyTensor = packOp.getSource().getDefiningOp<tensor::EmptyOp>();
    if (!emptyTensor)
      return failure();
    rewriter.replaceOp(packOp, packOp.getDest());
    return success();
  }
};

// If all the tiled dimension create unit tile loops pack can be rewritten as
// a reshape.
struct PackAsReshape : public OpRewritePattern<tensor::PackOp> {
  using OpRewritePattern<tensor::PackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PackOp packOp,
                                PatternRewriter &rewriter) const override {
    if (packOp.getPaddingValue())
      return failure();
    RankedTensorType sourceType = packOp.getSourceType();
    ArrayRef<int64_t> sourceShape = sourceType.getShape();
    size_t dimsToDrop = 0;
    for (auto [dim, tile] : packOp.getDimAndTileMapping()) {
      auto constantTile = getConstantIntValue(tile);
      if (constantTile && sourceShape[dim] == *constantTile)
        dimsToDrop++;
    }
    // All the tiled dimension need to be dropped, we cannot drop
    // single dimensions.
    if (dimsToDrop != sourceShape.size())
      return failure();

    auto reassoc = getReassociationIndicesForReshape(packOp.getSourceType(),
                                                     packOp.getDestType());
    if (!reassoc)
      return failure();
    Value expanded = linalgx::utils::expand(
        rewriter, packOp.getLoc(), packOp.getSource(), packOp.getDestType(),
        getReassociationIndicesAttribute(rewriter, *reassoc));
    rewriter.replaceOp(packOp, expanded);
    return success();
  }
};

// Fold: expand_shape(tensor.pack).
struct PackOfReshape : public OpRewritePattern<tensor::PackOp> {
  using OpRewritePattern<tensor::PackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PackOp packOp,
                                PatternRewriter &rewriter) const override {
    if (!packOp.getOuterDimsPerm().empty())
      return failure();

    auto expandShapeOp =
        packOp.getSource().getDefiningOp<tensor::ExpandShapeOp>();
    if (!expandShapeOp)
      return failure();
    SmallVector<ReassociationIndices, 4> reassoc =
        expandShapeOp.getReassociationIndices();
    // The operation must expand only leading dimensions.
    for (int i = reassoc.size() - 1; i > 0; i--)
      if (reassoc[i].size() != 1)
        return failure();

    // Shift the innerDimsPos by the number of leading ones.
    SmallVector<int64_t> innerDimsPos =
        llvm::to_vector(packOp.getInnerDimsPos());
    int64_t shift = reassoc[0].size() - 1;
    for (size_t i = 0; i < innerDimsPos.size(); i++)
      innerDimsPos[i] = innerDimsPos[i] - shift;

    auto newPackType = tensor::PackOp::inferPackedType(
        expandShapeOp.getSrcType(), packOp.getStaticInnerTiles(), innerDimsPos,
        packOp.getOuterDimsPerm());
    auto reassocExpand =
        getReassociationIndicesForReshape(newPackType, packOp.getDestType());
    if (!reassocExpand)
      return failure();

    Value destTensor = tensor::PackOp::createDestinationTensor(
        rewriter, packOp.getLoc(), expandShapeOp.getSrc(),
        packOp.getMixedTiles(), innerDimsPos, packOp.getOuterDimsPerm());
    Value packedVal = rewriter.create<tensor::PackOp>(
        packOp.getLoc(), expandShapeOp.getSrc(), destTensor, innerDimsPos,
        packOp.getMixedTiles(), packOp.getPaddingValue(),
        packOp.getOuterDimsPerm());

    Value expanded = linalgx::utils::expand(
        rewriter, packOp.getLoc(), packedVal, packOp.getDestType(),
        getReassociationIndicesAttribute(rewriter, *reassocExpand));
    rewriter.replaceOp(packOp, expanded);

    return success();
  }
};

struct FoldExpandShapeInParallelInsertOp
    : public OpRewritePattern<tensor::ParallelInsertSliceOp> {
  using OpRewritePattern<tensor::ParallelInsertSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ParallelInsertSliceOp insertOp,
                                PatternRewriter &rewriter) const override {
    auto expandShapeOp =
        insertOp.getSource().getDefiningOp<tensor::ExpandShapeOp>();
    if (!expandShapeOp)
      return failure();
    SliceVerificationResult res = isRankReducedType(
        expandShapeOp.getResultType(), expandShapeOp.getSrcType());
    if (res != SliceVerificationResult::Success)
      return failure();
    rewriter.updateRootInPlace(insertOp, [&]() {
      insertOp.setOperand(/*source=*/0, expandShapeOp.getSrc());
    });
    return success();
  }
};

struct SimplifyAndCanonicalizePack
    : public SimplifyAndCanonicalizePackBase<SimplifyAndCanonicalizePack> {
  void runOnOperation() override {
    MLIRContext *ctx = getOperation().getContext();
    RewritePatternSet patterns(ctx);
    tensor::populateSimplifyTensorPack(patterns);
    tensor::PackOp::getCanonicalizationPatterns(patterns, ctx);
    tensor::UnPackOp::getCanonicalizationPatterns(patterns, ctx);
    linalg::FillOp::getCanonicalizationPatterns(patterns, ctx);
    patterns.add<SimplifyPackToEmpty, PackAsReshape, PackOfReshape,
                 FoldExpandShapeInParallelInsertOp>(ctx);
    tensor::populateReassociativeReshapeFoldingPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // end namespace

void mlir::tpp::populateSinkPackPatterns(RewritePatternSet &patterns) {
  linalg::populateDataLayoutPropagationPatterns(
      patterns, [](Operation *op) { return true; });
  patterns.add<BubbleUpThroughFillOp>(patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createPackMatmulPass(ArrayRef<int64_t> blockingFactors) {
  return std::make_unique<PackMatmul>(blockingFactors);
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createPackConv2DNchwFchwPass(ArrayRef<int64_t> blockingFactors) {
  return std::make_unique<PackConv2DNchwFchw>(blockingFactors);
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createPackConv2DNhwcHwcfPass(ArrayRef<int64_t> blockingFactors) {
  return std::make_unique<PackConv2DNhwcHwcf>(blockingFactors);
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::tpp::createPackVNNIPass() {
  return std::make_unique<PackVNNI>();
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createPropagatePackUnPackPass() {
  return std::make_unique<PropagatePackUnPack>();
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createSimplifyAndCanonicalizePackPass() {
  return std::make_unique<SimplifyAndCanonicalizePack>();
}
