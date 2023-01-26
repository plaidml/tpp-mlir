//===- ConvertToBlockLayoutAndBack.cpp ---------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Tpp/TppUtils.h"
#include "TPP/Dialect/VNNI/VNNIOps.h"
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
#include "mlir/Dialect/Traits.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TPP/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

static std::optional<int64_t> getConstantRange(const Range &range) {
  std::optional<int64_t> stride = getConstantIntValue(range.stride);
  if (!stride || *stride != 1)
    return std::nullopt;
  std::optional<int64_t> offset = getConstantIntValue(range.offset);
  if (!offset)
    return std::nullopt;
  std::optional<int64_t> size = getConstantIntValue(range.size);
  if (!size)
    return std::nullopt;
  return (*size - *offset);
}

static bool isFullTile(int64_t tileFactor, int64_t range) {
  return range % tileFactor == 0;
}

// Statically validate the 'tile' along the dimension 'dim'. If the tile or the
// dimension are not-statically known return true, as no assumption can be made.
// If the the tile and the dimension are statically known require the tile to be
// a full tile.
static bool validateFullTilesOnDim(linalg::LinalgOp linalgOp,
                                   const OpFoldResult &tile, size_t dim) {
  OpBuilder builder(linalgOp);
  SmallVector<Range> iterationDomain =
      cast<TilingInterface>(linalgOp.getOperation())
          .getIterationDomain(builder);
  if (dim >= iterationDomain.size())
    return false;

  auto tileFactor = getConstantIntValue(tile);
  auto rangeOnDim = getConstantRange(iterationDomain[dim]);

  // If the tile factor or the range are non-constant, the tile size is
  // considered to be valid.
  if (!tileFactor || !rangeOnDim)
    return true;

  // Tiling with '0' along 'dim' is valid - no tiling.
  if (*tileFactor == 0)
    return true;

  return isFullTile(*tileFactor, *rangeOnDim);
}

// TODO: Expose this as tile utils (it should be used by fusion too).
static bool validateFullTilesOnDims(linalg::LinalgOp linalgOp,
                                    ArrayRef<OpFoldResult> tiles,
                                    ArrayRef<size_t> dims) {
  if (dims.size() != tiles.size())
    return false;
  size_t idxInTiles = 0;
  for (size_t dim : dims) {
    if (!validateFullTilesOnDim(linalgOp, tiles[idxInTiles++], dim))
      return false;
  }
  return true;
}

// Helper function to create the pack operation.
static Value toPackLayoutImpl(OpBuilder &builder, Location loc, Value input,
                              ArrayRef<OpFoldResult> tiles,
                              ArrayRef<int64_t> innerDimsPos,
                              ArrayRef<int64_t> outerDimsPerm) {
  SmallVector<Value> dynamicTiles;
  SmallVector<int64_t> staticTiles;
  dispatchIndexOpFoldResults(tiles, dynamicTiles, staticTiles);
  ShapedType result = tensor::PackOp::inferPackedType(
      input.getType(), staticTiles, innerDimsPos, outerDimsPerm);
  ShapedType inputType = input.getType().cast<ShapedType>();
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
  return builder.create<tensor::UnPackOp>(loc, input, output, innerDimPos,
                                          tiles, outerDimsPerm);
}

static Value handleLayoutNC_NCnc(OpBuilder &builder, Location loc, Value input,
                                 Value output, ArrayRef<OpFoldResult> tiles) {
  assert(tiles.size() == 2 && "expect two tile sizes for NC_NCnc");
  SmallVector<int64_t> innerDimPos = {0, 1};
  if (!output)
    return toPackLayoutImpl(builder, loc, input, tiles, innerDimPos,
                            /*outerDimsPerm=*/{});
  return toUnPackLayoutImpl(builder, loc, input, output, tiles, innerDimPos,
                            /*outerDimsPerm=*/{});
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
// Helper function to pack from NC to NCnc.
static Value toPackLayoutNC_NCnc(OpBuilder &builder, Location loc, Value input,
                                 ArrayRef<OpFoldResult> tiles) {
  return handleLayoutNC_NCnc(builder, loc, input, nullptr, tiles);
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

// Helper function to unpack from NCnc to NC.
static Value fromPackLayoutNCnc_NC(OpBuilder &builder, Location loc,
                                   Value input, Value output,
                                   ArrayRef<OpFoldResult> tiles) {
  return handleLayoutNC_NCnc(builder, loc, input, output, tiles);
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

// Helper function to pack from KC to CKkc.
static Value toPackLayoutKC_CKkc(OpBuilder &builder, Location loc, Value input,
                                 ArrayRef<OpFoldResult> tiles) {
  assert(tiles.size() == 2 && "expect two tiles size for KC_CKkc");
  SmallVector<int64_t> innerDimPos = {0, 1};
  SmallVector<int64_t> outerDimPerm = {1, 0};
  return toPackLayoutImpl(builder, loc, input, tiles, innerDimPos,
                          outerDimPerm);
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

  SmallVector<Value> inputOperands = convOp.getDpsInputOperands();
  SmallVector<Value> outputOperands = convOp.getDpsInitOperands();

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

static FailureOr<linalg::GenericOp>
packMatmulOpImpl(RewriterBase &rewriter, linalg::MatmulOp matmulOp,
                 ArrayRef<OpFoldResult> tiles) {

  if (matmulOp.hasDynamicShape())
    return rewriter.notifyMatchFailure(matmulOp, "require static shape");

  if (matmulOp.hasBufferSemantics())
    return rewriter.notifyMatchFailure(matmulOp, "require tensor semantics");

  OpFoldResult tileOnI = tiles[0];
  OpFoldResult tileOnJ = tiles[1];
  OpFoldResult tileOnK = tiles[2];
  if (!validateFullTilesOnDims(matmulOp, {tileOnI, tileOnJ, tileOnK},
                               {/*I=*/0, /*J=*/1, /*K=*/2}))
    return rewriter.notifyMatchFailure(matmulOp, "expect full tiles only");
  SmallVector<OpFoldResult, 2> tilesOnA = {tileOnI, tileOnK};
  SmallVector<OpFoldResult, 2> tilesOnB = {tileOnK, tileOnJ};
  SmallVector<OpFoldResult, 2> tilesOnC = {tileOnI, tileOnJ};

  Location loc = matmulOp.getLoc();
  // reshape input A and B.
  Value packedMatrixA =
      toPackLayoutNC_NCnc(rewriter, loc, matmulOp.getInputs()[0], tilesOnA);
  Value packedMatrixB =
      toPackLayoutKC_CKkc(rewriter, loc, matmulOp.getInputs()[1], tilesOnB);
  SmallVector<Value> packedInputs = {packedMatrixA, packedMatrixB};

  // reshape output C.
  Value packMatrixC =
      toPackLayoutNC_NCnc(rewriter, loc, matmulOp.getOutputs()[0], tilesOnC);

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
      ArrayRef<utils::IteratorType>{
          utils::IteratorType::parallel, utils::IteratorType::parallel,
          utils::IteratorType::reduction, utils::IteratorType::parallel,
          utils::IteratorType::parallel, utils::IteratorType::reduction},
      /*doc=*/"", /*libraryCall=*/"");
  rewriter.inlineRegionBefore(matmulOp.getRegion(), replacementOp.getRegion(),
                              replacementOp.getRegion().begin());

  // convert back from pack layout.
  Value outPackTensor = replacementOp.getResult(0);
  Value outUnPackTensor = matmulOp.getOutputs()[0];
  Value outReplacement = fromPackLayoutNCnc_NC(rewriter, loc, outPackTensor,
                                               outUnPackTensor, tilesOnC);
  rewriter.replaceOp(matmulOp, outReplacement);
  return replacementOp;
}

bool isVNNIPacked(linalg::GenericOp matmulOp) {
  // TODO add VNNI packing checks here
  auto indexingMap = matmulOp.getIndexingMapsArray()[1];
  return indexingMap.getNumDims() == 7;
}

bool isMatmulOp(linalg::GenericOp matmulOp) {
  // TODO check structural and access pattern.
  return tpp::utils::hasMatmulBody(matmulOp);
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

  return packMatmulOpImpl(rewriter, matmulOp, tiles);
}

FailureOr<linalg::GenericOp>
mlir::linalgx::packVNNIMatmulOp(RewriterBase &rewriter,
                                linalg::GenericOp matmulOp) {
  if (matmulOp.getInputs().size() > 0 &&
      !vnni::utils::isBF16Type(matmulOp.getInputs()[0].getType()))
    return rewriter.notifyMatchFailure(matmulOp, "require bf16 type");

  if (matmulOp.hasDynamicShape())
    return rewriter.notifyMatchFailure(matmulOp, "require static shape");

  if (matmulOp.hasBufferSemantics())
    return rewriter.notifyMatchFailure(matmulOp, "require tensor semantics");

  if (!isMatmulOp(matmulOp))
    return rewriter.notifyMatchFailure(matmulOp, "require matmul semantics");

  if (isVNNIPacked(matmulOp))
    return rewriter.notifyMatchFailure(matmulOp, "already packed to VNNI");

  Location loc = matmulOp.getLoc();
  auto blockingFactor =
      vnni::utils::getVNNIBlockingFactor(matmulOp.getInputs()[1].getType());
  if (!blockingFactor)
    return rewriter.notifyMatchFailure(matmulOp,
                                       "unsupported blocking factor for type");
  OpFoldResult tileOnI = rewriter.getI64IntegerAttr(*blockingFactor);
  SmallVector<OpFoldResult, 1> tilesOnB = {tileOnI};
  // reshape input B.
  Value packedMatrixB =
      toPackLayout_VNNI(rewriter, loc, matmulOp.getInputs()[1], tilesOnB);
  MLIRContext *ctx = matmulOp.getContext();
  AffineExpr p1, p2, r1, p3, p4, r2, r3;
  bindDims(ctx, p1, p2, r1, p3, p4, r2, r3);
  SmallVector<Value> packedInputs = {matmulOp.getInputs()[0], packedMatrixB};
  AffineMap mapA =
      AffineMap::get(/*dims=*/7, /*symbols=*/0, {p1, r1, p3, r2}, ctx);
  AffineMap mapB =
      AffineMap::get(/*dims=*/7, /*symbols=*/0,
                     {p2, r1, r2.floorDiv(*blockingFactor), p4, r3}, ctx);
  AffineMap mapC =
      AffineMap::get(/*dims=*/7, /*symbols=*/0, {p1, p2, p3, p4}, ctx);
  Value matrixC = matmulOp.getOutputs()[0];
  linalg::GenericOp replacementOp = rewriter.create<linalg::GenericOp>(
      loc, matrixC.getType(), packedInputs, ValueRange{matrixC},
      ArrayRef<AffineMap>{mapA, mapB, mapC},
      ArrayRef<mlir::utils::IteratorType>{mlir::utils::IteratorType::parallel,
                                          mlir::utils::IteratorType::parallel,
                                          mlir::utils::IteratorType::reduction,
                                          mlir::utils::IteratorType::parallel,
                                          mlir::utils::IteratorType::parallel,
                                          mlir::utils::IteratorType::reduction,
                                          mlir::utils::IteratorType::reduction},
      /*doc=*/"", /*libraryCall=*/"");
  rewriter.inlineRegionBefore(matmulOp.getRegion(), replacementOp.getRegion(),
                              replacementOp.getRegion().begin());

  rewriter.replaceOp(matmulOp, replacementOp.getResult(0));
  return replacementOp;
}

FailureOr<vnni::BRGemmOp>
mlir::linalgx::packVNNIBRGemmOp(RewriterBase &rewriter,
                                linalg::BatchReduceMatmulOp brgemmOp) {
  if (!vnni::utils::isBF16Type(brgemmOp.getInputs()[0].getType()))
    return rewriter.notifyMatchFailure(brgemmOp, "require bf16 type");

  if (brgemmOp.hasDynamicShape())
    return rewriter.notifyMatchFailure(brgemmOp, "require static shape");

  if (brgemmOp.hasBufferSemantics())
    return rewriter.notifyMatchFailure(brgemmOp, "require tensor semantics");

  assert(vnni::utils::isBF16Type(brgemmOp.getInputs()[0].getType()));
  // Set blocking factor to size 2
  OpFoldResult tileOnI = rewriter.getI64IntegerAttr(2);
  SmallVector<OpFoldResult, 1> tilesOnB = {tileOnI};

  Location loc = brgemmOp.getLoc();
  // reshape input B.
  Value packedMatrixB =
      toPackBRGemmLayout_VNNI(rewriter, loc, brgemmOp.getInputs()[1], tilesOnB);
  auto replacementOp = rewriter.create<vnni::BRGemmOp>(
      loc, brgemmOp.getOutputs()[0].getType(), brgemmOp.getInputs()[0],
      packedMatrixB, brgemmOp.getOutputs()[0]);
  rewriter.replaceOp(brgemmOp, replacementOp.getResult(0));
  return replacementOp;
}
namespace {

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
    tensor::UnPackOp unpackOp = inputPad.getDefiningOp<tensor::UnPackOp>();
    if (!unpackOp)
      return failure();

    // bail out if one of the padded dimension is a tiled one.
    llvm::SmallBitVector paddedDims = padOp.getPaddedDims();
    ArrayRef<int64_t> innerDimsPos = unpackOp.getInnerDimsPos();
    llvm::SmallBitVector innerDims(paddedDims.size());
    for (int64_t dim : innerDimsPos)
      paddedDims.flip(dim);
    if (paddedDims.anyCommon(innerDims))
      return failure();

    ArrayRef<int64_t> outerDimsPerm = unpackOp.getOuterDimsPerm();
    SmallVector<OpFoldResult> lowPad = padOp.getMixedLowPad();
    SmallVector<OpFoldResult> highPad = padOp.getMixedHighPad();
    if (!outerDimsPerm.empty()) {
      lowPad = interchange<OpFoldResult>(lowPad, outerDimsPerm);
      highPad = interchange<OpFoldResult>(highPad, outerDimsPerm);
    }
    size_t innerDimsPosSize = innerDimsPos.size();
    lowPad.append(innerDimsPosSize, rewriter.getIndexAttr(0));
    highPad.append(innerDimsPosSize, rewriter.getIndexAttr(0));

    auto newPadOp = rewriter.create<tensor::PadOp>(
        padOp.getLoc(), /*result type=*/nullptr, unpackOp.getSource(), lowPad,
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
        rewriter, padOp.getLoc(), padOpRes, outputUnPack,
        unpackOp.getMixedTiles(), innerDimsPos, unpackOp.getOuterDimsPerm());

    rewriter.replaceOp(padOp, replacement);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

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
    if (!tpp::utils::isMarkedWithTpp(linalgOp, "tpp.matmul"))
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
    patterns.add<DoItOnConv2DNhwcHwcf>(ctx, blockingFactors);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
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
    FailureOr<vnni::BRGemmOp> packedBRGemm =
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
    mlir::tpp::populateSinkPackPatterns(patterns);
    mlir::tensor::populateSimplifyTensorPack(patterns);
    patterns.add<VNNIOnMatmul>(ctx);
    patterns.add<VNNIOnBRGemm>(ctx);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

struct PropagatePackUnPack
    : public PropagatePackUnPackBase<PropagatePackUnPack> {
  void runOnOperation() override {
    MLIRContext *ctx = getOperation().getContext();
    RewritePatternSet patterns(ctx);
    tpp::populateSinkPackPatterns(patterns);
    tensor::populateSimplifyTensorPack(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // end namespace

void mlir::tpp::populateSinkPackPatterns(RewritePatternSet &patterns) {
  linalg::populateDataLayoutPropagationPatterns(patterns);
  patterns.add<PropagateThroughPadOp>(patterns.getContext());
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

std::unique_ptr<OperationPass<func::FuncOp>> mlir::tpp::createPackVNNIPass() {
  return std::make_unique<PackVNNI>();
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createPropagatePackUnPackPass() {
  return std::make_unique<PropagatePackUnPack>();
}
