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
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::linalgx;

#define GEN_PASS_CLASSES
#include "Standalone/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

static DenseMap<int64_t, OpFoldResult>
buildTileAndPosMapping(ArrayRef<int64_t> dimPos, ArrayRef<OpFoldResult> tiles) {
  DenseMap<int64_t, OpFoldResult> tileAndPosMapping;
  int64_t dims = dimPos.size();
  for (int64_t idx = 0; idx < dims; idx++)
    tileAndPosMapping[dimPos[idx]] = tiles[idx];
  return tileAndPosMapping;
}

/// Interchange `elements` starting at offset `offset` based on the indexes in
/// `interchangeVector`.
// TODO: (lorenzo) avoid copy and paste
template <typename T>
static SmallVector<T> interchange(ArrayRef<T> elements,
                                  ArrayRef<int64_t> interchangeVector,
                                  int64_t offset) {
  SmallVector<T> rearrangedElements = llvm::to_vector(elements);
  if (interchangeVector.empty())
    return rearrangedElements;
  // assert((rearrangedElements.size() - offset) == interchangeVector.size() &&
  //        "number of elements must equal number of permutations");
  for (int64_t idx = 0, end = interchangeVector.size(); idx < end; idx++) {
    rearrangedElements[interchangeVector[idx] + offset] =
        elements[idx + offset];
  }
  return rearrangedElements;
}

/// Infer the packed type.
// TODO: (lorenzo) avoid copy and paste.
static ShapedType
inferPackedType(ShapedType sourceType, ArrayRef<int64_t> innerTiles,
                const DenseMap<int64_t, OpFoldResult> &tileAndPosMapping,
                ArrayRef<int64_t> outerDimsPos) {
  SmallVector<int64_t> inferredShape;
  int64_t rank = sourceType.getRank();

  // tile loop.
  for (auto dim : llvm::seq<int64_t>(0, rank)) {
    if (tileAndPosMapping.count(dim)) {
      Optional<int64_t> tileSize =
          getConstantIntValue(tileAndPosMapping.lookup(dim));
      if (sourceType.isDynamicDim(dim) || !tileSize) {
        inferredShape.push_back(ShapedType::kDynamicSize);
      } else {
        int64_t sizeTiledDim = ceilDiv(sourceType.getDimSize(dim), *tileSize);
        inferredShape.push_back(sizeTiledDim);
      }
    } else {
      inferredShape.push_back(sourceType.getShape()[dim]);
    }
  }

  // swap tile loops if `outer_dims_pos` is available.
  inferredShape =
      interchange<int64_t>(inferredShape, outerDimsPos, /*offset=*/0);

  // point loop.
  inferredShape.append(innerTiles.begin(), innerTiles.end());

  return TypeSwitch<Type, ShapedType>(sourceType)
      .Case<RankedTensorType>([&](RankedTensorType t) -> ShapedType {
        return RankedTensorType::get(inferredShape,
                                     sourceType.getElementType());
      })
      .Case<MemRefType>([&](MemRefType t) -> ShapedType {
        return MemRefType::get(inferredShape, sourceType.getElementType());
      })
      .Default([&](Type t) {
        llvm_unreachable("unexpected type");
        return nullptr;
      });
}

/// Helper function to create the pack operation.
static Value toPackLayoutImpl(Location loc, Value input,
                              ArrayRef<OpFoldResult> tiles,
                              ArrayRef<int64_t> innerDimPos,
                              ArrayRef<int64_t> outerDimPerm,
                              OpBuilder &builder, bool useAlloc = false) {
  DenseMap<int64_t, OpFoldResult> tileAndPosMapping =
      buildTileAndPosMapping(innerDimPos, tiles);
  SmallVector<Value> dynamicTiles;
  SmallVector<int64_t> staticTiles;
  dispatchIndexOpFoldResults(tiles, dynamicTiles, staticTiles,
                             ShapedType::kDynamicSize);
  ShapedType result = inferPackedType(input.getType(), staticTiles,
                                      tileAndPosMapping, outerDimPerm);
  ShapedType inputType = input.getType().cast<ShapedType>();
  ArrayRef<int64_t> shape = result.getShape();
  Value output;
  if (useAlloc)
    output = builder.create<bufferization::AllocTensorOp>(
        loc, RankedTensorType::get(shape, inputType.getElementType()),
        ValueRange{});
  else
    output = builder.create<linalg::InitTensorOp>(loc, shape,
                                                  inputType.getElementType());
  return builder
      .create<linalgx::PackOp>(loc, input, output, innerDimPos, outerDimPerm,
                               tiles)
      .getResults()[0];
}

/// Helper function to create the unpack operation.
static Value toUnPackLayoutImpl(Location loc, Value input, Value output,
                                ArrayRef<OpFoldResult> tiles,
                                ArrayRef<int64_t> innerDimPos,
                                OpBuilder &builder) {
  return builder
      .create<linalgx::UnPackOp>(loc, input, output, innerDimPos, tiles)
      .getResults()[0];
}

/// Helper function to get an NCnc pack layout.
static Value toPackLayoutNCnc(Location loc, Value input,
                              ArrayRef<OpFoldResult> tiles, OpBuilder &builder,
                              bool useAlloc = false) {
  assert(tiles.size() == 2 && "expect two tile sizes for NCnc");
  SmallVector<int64_t> innerDimPos = {0, 1};
  return toPackLayoutImpl(loc, input, tiles, innerDimPos, {}, builder,
                          useAlloc);
}

/// Helper function to get a CKkc pack layout.
static Value toPackLayoutCKkc(Location loc, Value input,
                              ArrayRef<OpFoldResult> tiles, OpBuilder &builder,
                              bool useAlloc = false) {
  assert(tiles.size() == 2 && "expect two tiles size for CKkc");
  SmallVector<int64_t> innerDimPos = {0, 1};
  SmallVector<int64_t> outerDimPerm = {1, 0};
  return toPackLayoutImpl(loc, input, tiles, innerDimPos, outerDimPerm, builder,
                          useAlloc);
}

/// Helper function to get a NCHWc pack layout.
static Value toPackLayoutNCHWc(Location loc, Value input,
                               ArrayRef<OpFoldResult> tiles,
                               OpBuilder &builder) {
  assert(tiles.size() == 1 && "expect one tile size for NCHWc");
  SmallVector<int64_t> innerDimPos = {1};
  return toPackLayoutImpl(loc, input, tiles, innerDimPos, {}, builder);
}

/// Helper function to get a KCRSck pack layout.
static Value toPackLayoutKCRSck(Location loc, Value input,
                                ArrayRef<OpFoldResult> tiles,
                                OpBuilder &builder) {
  assert(tiles.size() == 2 && "expect two tiles size for KCRSck");
  SmallVector<int64_t> innerDimPos = {1, 0};
  return toPackLayoutImpl(loc, input, tiles, innerDimPos, {}, builder);
}

/// Helper function to get an NC unpack layout from NCnc.
static Value fromPackLayoutNCnc(Location loc, Value input, Value output,
                                ArrayRef<OpFoldResult> tiles,
                                OpBuilder &builder) {
  assert(tiles.size() == 2 && "expect two tile sizes for NCnc");
  SmallVector<int64_t> innerDimPos = {0, 1};
  return toUnPackLayoutImpl(loc, input, output, tiles, innerDimPos, builder);
}

/// Helper function to get an NCHW unpack layout from NCHWc.
static Value fromPackLayoutNCHWc(Location loc, Value input, Value output,
                                 ArrayRef<OpFoldResult> tiles,
                                 OpBuilder &builder) {
  assert(tiles.size() == 1 && "expect one tile size for NCHWc");
  SmallVector<int64_t> innerDimPos = {1};
  return toUnPackLayoutImpl(loc, input, output, tiles, innerDimPos, builder);
}

/// Helper function ensure packing preconditions. We pack only if the operation
/// has static shape and it is at tensor level.
static LogicalResult BlockOpPreconditions(linalg::LinalgOp linalgOp) {
  if (linalgOp.hasDynamicShape() || linalgOp.hasBufferSemantics())
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// Conv2DNchwFchwOp
//===----------------------------------------------------------------------===//

FailureOr<linalg::GenericOp>
mlir::linalgx::blockConv2DNchwFchwOp(RewriterBase &rewriter,
                                     linalg::Conv2DNchwFchwOp convOp,
                                     ArrayRef<OpFoldResult> tiles) {
  if ((tiles.size() != 2) || (failed(BlockOpPreconditions(convOp))))
    return failure();

  Location loc = convOp.getLoc();
  MLIRContext *ctx = convOp.getContext();
  SmallVector<Value, 2> reshapedInputTensors;

  SmallVector<Value> inputOperands = convOp.getInputOperands();
  SmallVector<Value> outputOperands = convOp.getOutputOperands();

  // pack the image and the filter.
  Value image = inputOperands[0];
  Value packedImage = toPackLayoutNCHWc(loc, image, tiles[0], rewriter);
  Value filter = inputOperands[1];
  Value packedFilter = toPackLayoutKCRSck(loc, filter, tiles, rewriter);
  SmallVector<Value> packedInputs = {packedImage, packedFilter};

  // pack the output.
  Value output = outputOperands[0];
  Value packedOutput = toPackLayoutNCHWc(loc, output, tiles[0], rewriter);

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

  // convert back from block layout.
  Value outPackedTensor = replacementOp.getResult(0);
  Value outUnPackedTensor = outputOperands[0];
  Value outReplacement = fromPackLayoutNCHWc(
      loc, outPackedTensor, outUnPackedTensor, tiles[0], rewriter);
  rewriter.replaceOp(convOp, outReplacement);
  return replacementOp;
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
mlir::linalgx::blockMatmulOp(RewriterBase &rewriter, linalg::MatmulOp matmulOp,
                             ArrayRef<OpFoldResult> tiles) {
  if ((tiles.size() != 3) || (failed(BlockOpPreconditions(matmulOp))))
    return failure();

  OpFoldResult tileOnI = tiles[0];
  OpFoldResult tileOnJ = tiles[1];
  OpFoldResult tileOnK = tiles[2];
  SmallVector<OpFoldResult, 2> tilesOnA = {tileOnI, tileOnK};
  SmallVector<OpFoldResult, 2> tilesOnB = {tileOnK, tileOnJ};
  SmallVector<OpFoldResult, 2> tilesOnC = {tileOnI, tileOnJ};

  Location loc = matmulOp.getLoc();
  SmallVector<Value> reshapedInputTensors;
  // reshape input A and B
  Value packedMatrixA =
      toPackLayoutNCnc(loc, matmulOp.getInputs()[0], tilesOnA, rewriter, true);
  Value packedMatrixB =
      toPackLayoutCKkc(loc, matmulOp.getInputs()[1], tilesOnB, rewriter, true);
  SmallVector<Value> packedInputs = {packedMatrixA, packedMatrixB};

  // reshape output C.
  Value packMatrixC =
      toPackLayoutNCnc(loc, matmulOp.getOutputs()[0], tilesOnC, rewriter, true);

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
  Value outReplacement = fromPackLayoutNCnc(loc, outPackTensor, outUnPackTensor,
                                            tilesOnC, rewriter);
  rewriter.replaceOp(matmulOp, outReplacement);
  return replacementOp;
}

namespace {

// Pack MatmulOp.
struct DoItOnMatmul : public OpRewritePattern<linalg::MatmulOp> {
  DoItOnMatmul(MLIRContext *context, ArrayRef<int64_t> tiles,
               PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::MatmulOp>(context, benefit), tiles(tiles) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    FailureOr<linalg::GenericOp> blockedMatmul = mlir::linalgx::blockMatmulOp(
        rewriter, matmulOp, getAsOpFoldResult(rewriter.getI64ArrayAttr(tiles)));
    if (failed(blockedMatmul))
      return failure();
    return success();
  }

private:
  ArrayRef<int64_t> tiles;
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
    SmallVector<Value> inputOperands = linalgOp.getInputOperands();
    SmallVector<Value> outputOperands = linalgOp.getOutputOperands();
    rewriter.replaceOpWithNewOp<linalg::MatmulOp>(
        linalgOp, linalgOp.getResultTypes(), inputOperands, outputOperands);
    return success();
  }
};

// Entry point for packing a matmul operation.
// Pack MatmulOp as following:
// [NB][KB][nb][kb] += [NB][CB][nb][cb] * [KB][CB][cb][kb]
// CB = batch reduce dimension.
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
    mlir::tpp::populateSinkRelayoutPatterns(patterns);
    patterns.add<DoItOnMatmul>(ctx, blockingFactors);
    patterns.add<DeGeneralizeMatmul>(ctx);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

struct DoItOnConv2DNchwFchw
    : public OpRewritePattern<linalg::Conv2DNchwFchwOp> {
  DoItOnConv2DNchwFchw(MLIRContext *context, ArrayRef<int64_t> tiles,
                       PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::Conv2DNchwFchwOp>(context, benefit),
        tiles(tiles) {}

  LogicalResult matchAndRewrite(linalg::Conv2DNchwFchwOp linalgOp,
                                PatternRewriter &rewriter) const override {
    FailureOr<linalg::GenericOp> maybeGeneric =
        mlir::linalgx::blockConv2DNchwFchwOp(
            rewriter, linalgOp,
            getAsOpFoldResult(rewriter.getI64ArrayAttr(tiles)));
    if (failed(maybeGeneric))
      return failure();
    return success();
  }

private:
  ArrayRef<int64_t> tiles;
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

//===----------------------------------------------------------------------===//
// PropagateThrElementWiseOp
//===----------------------------------------------------------------------===//

struct PropagateThrElementWiseOp : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  // Check sinking preconditions: a) all loops mut
  // be parallel.
  LogicalResult checkPreconditions(linalg::GenericOp linalgOp) const {
    if (linalgOp.getNumLoops() != linalgOp.getNumParallelLoops())
      return failure();

    return success();
  }

  Value getPackOperand(OpOperand *operand, linalg::GenericOp linalgOp,
                       DenseMap<int64_t, OpFoldResult> dimAndTileMapping,
                       PatternRewriter &rewriter) const {
    linalgx::UnPackOp unpackOp =
        operand->get().getDefiningOp<linalgx::UnPackOp>();
    SmallVector<OpFoldResult> tiles;
    SmallVector<int64_t> innerDimsPos;
    // If the operand comes from an unpack operation simply pack the operand
    // with the same tiles, and dimsPos extracted from the unpack, otherwise
    // infer them from `dimsAndTileMapping`.
    if (unpackOp) {
      tiles = unpackOp.getMixedTiles();
      innerDimsPos = extractFromI64ArrayAttr(unpackOp.getInnerDimsPos());
    } else {
      AffineMap mapOperand = linalgOp.getMatchingIndexingMap(operand);
      for (unsigned pos = 0; pos < mapOperand.getNumResults(); pos++) {
        unsigned posInDomain = mapOperand.getDimPosition(pos);
        if (dimAndTileMapping.count(posInDomain)) {
          tiles.push_back(dimAndTileMapping[posInDomain]);
          innerDimsPos.push_back(pos);
        }
      }
    }
    return toPackLayoutImpl(linalgOp.getLoc(), operand->get(), tiles,
                            innerDimsPos, {}, rewriter);
  }

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (failed(checkPreconditions(linalgOp)))
      return failure();

    // Pack and unpack operate on result of each operand map in the linalg
    // operation. We need to map these dimensions (co-domain) to the domain of
    // the linalg operation. Scan each input and output operands. For each map
    // associated to the operand check the equivalent dimension in the domain
    // and bind it with the tile size.
    DenseMap<int64_t, OpFoldResult> dimAndTileMapping;
    for (OpOperand *operand : linalgOp.getInputAndOutputOperands()) {
      linalgx::UnPackOp unpackOp =
          operand->get().getDefiningOp<linalgx::UnPackOp>();
      if (!unpackOp)
        continue;
      // avoid having to deal with tile loop interchange.
      SmallVector<int64_t> outerDimsPerm =
          extractFromI64ArrayAttr(unpackOp.getOuterDimsPos());
      if (!outerDimsPerm.empty())
        return failure();
      // map *domain* of linalg operation to tiles.
      DenseMap<int64_t, OpFoldResult> currentDimAndTileMapping =
          unpackOp.getDimAndTileMapping();
      AffineMap mapOperand = linalgOp.getMatchingIndexingMap(operand);
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
    if (dimAndTileMapping.empty())
      return failure();

    SmallVector<Value> packedInputOperands;
    for (OpOperand *operand : linalgOp.getInputOperands()) {
      Value packedOperand =
          getPackOperand(operand, linalgOp, dimAndTileMapping, rewriter);
      packedInputOperands.push_back(packedOperand);
    }

    SmallVector<Value> packedOutputOperands;
    SmallVector<Type> packedOutputTypes;
    SmallVector<Value> unpackOutputs;
    for (OpOperand *operand : linalgOp.getOutputOperands()) {
      Value packedOperand =
          getPackOperand(operand, linalgOp, dimAndTileMapping, rewriter);
      packedOutputOperands.push_back(packedOperand);
      packedOutputTypes.push_back(packedOperand.getType());
      linalgx::UnPackOp unpackOp =
          operand->get().getDefiningOp<linalgx::UnPackOp>();
      if (unpackOp)
        unpackOutputs.push_back(unpackOp.getOutput());
    }

    unsigned packedDims = dimAndTileMapping.size();
    SmallVector<AffineMap> newMaps;
    // get the new map for each operand.
    for (OpOperand *operand : linalgOp.getInputAndOutputOperands()) {
      AffineMap mapOperand = linalgOp.getMatchingIndexingMap(operand);
      unsigned numSymbols = 0;
      unsigned numDims = linalgOp.getNumLoops() + packedDims;
      SmallVector<AffineExpr> dimTiledLoops;
      SmallVector<AffineExpr> dimPointLoops;
      for (unsigned posInCodomain = 0;
           posInCodomain < mapOperand.getNumResults(); posInCodomain++) {
        unsigned posInDomain = mapOperand.getDimPosition(posInCodomain);
        if (dimAndTileMapping.count(posInDomain)) {
          dimTiledLoops.push_back(rewriter.getAffineDimExpr(posInDomain));
          dimPointLoops.push_back(
              rewriter.getAffineDimExpr(posInDomain + packedDims));
        }
      }
      dimTiledLoops.append(dimPointLoops.begin(), dimPointLoops.end());
      AffineMap newMap = AffineMap::get(numDims, numSymbols, dimTiledLoops,
                                        linalgOp.getContext());
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
    for (OpOperand *operand : replacementOp.getOutputOperands()) {
      linalgx::PackOp packOp = operand->get().getDefiningOp<linalgx::PackOp>();
      Value result = replacementOp.getTiedOpResult(operand);
      if (unpackOutputs.empty())
        outReplacements.push_back(toUnPackLayoutImpl(
            linalgOp.getLoc(), result, linalgOp.getOutputs()[idx++],
            packOp.getMixedTiles(),
            extractFromI64ArrayAttr(packOp.getInnerDimsPos()), rewriter));
      else
        outReplacements.push_back(toUnPackLayoutImpl(
            linalgOp.getLoc(), result, unpackOutputs[idx++],
            packOp.getMixedTiles(),
            extractFromI64ArrayAttr(packOp.getInnerDimsPos()), rewriter));
    }
    rewriter.replaceOp(linalgOp, outReplacements);
    return success();
  }
};

} // end namespace

void mlir::tpp::populateSinkRelayoutPatterns(RewritePatternSet &patterns) {
  patterns.add<PropagateThrElementWiseOp>(patterns.getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createBlockMatmulLayout() {
  return std::make_unique<BlockMatmulLayout>();
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createBlockConv2DNchwFchwLayout() {
  return std::make_unique<BlockConv2DNchwFchwLayout>();
}
