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
                              OpBuilder &builder) {
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
  Value output = builder.create<linalg::InitTensorOp>(
      loc, shape, inputType.getElementType());
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
                              ArrayRef<OpFoldResult> tiles,
                              OpBuilder &builder) {
  assert(tiles.size() == 2 && "expect two tile sizes for NCnc");
  SmallVector<int64_t> innerDimPos = {0, 1};
  return toPackLayoutImpl(loc, input, tiles, innerDimPos, {}, builder);
}

/// Helper function to get a CKkc pack layout.
static Value toPackLayoutCKkc(Location loc, Value input,
                              ArrayRef<OpFoldResult> tiles,
                              OpBuilder &builder) {
  assert(tiles.size() == 2 && "expect two tiles size for CKkc");
  SmallVector<int64_t> innerDimPos = {0, 1};
  SmallVector<int64_t> outerDimPerm = {1, 0};
  return toPackLayoutImpl(loc, input, tiles, innerDimPos, outerDimPerm,
                          builder);
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

FailureOr<linalg::GenericOp>
mlir::linalgx::blockMatmulOp(RewriterBase &rewriter, linalg::MatmulOp matmulOp,
                             ArrayRef<OpFoldResult> tiles) {
  if ((tiles.size() != 2) || (failed(BlockOpPreconditions(matmulOp))))
    return failure();

  Location loc = matmulOp.getLoc();
  SmallVector<Value, 2> reshapedInputTensors;
  // reshape input A to NCnc and B to CKkc.
  Value packedMatrixA =
      toPackLayoutNCnc(loc, matmulOp.getInputs()[0], tiles, rewriter);
  Value packedMatrixB =
      toPackLayoutCKkc(loc, matmulOp.getInputs()[1], tiles, rewriter);
  SmallVector<Value> packedInputs = {packedMatrixA, packedMatrixB};

  // reshape output C to NCnc.
  Value packMatrixC =
      toPackLayoutNCnc(loc, matmulOp.getOutputs()[0], tiles, rewriter);

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
  Value outReplacement =
      fromPackLayoutNCnc(loc, outPackTensor, outUnPackTensor, tiles, rewriter);
  rewriter.replaceOp(matmulOp, outReplacement);
  return replacementOp;
}

namespace {

// Relayout MatmulOp as following:
// [NB][KB][nb][kb] += [NB][CB][nb][cb] * [KB][CB][cb][kb]
// CB = batch reduce dimension.
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

#if 0
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
    //std::pair<AffineMap, AffineMap> maps = getMapsFromBlockLayoutNCnc_NC(
    //    /*dims=*/4, getBlockingFactors(fromBlockLayout), linalgOp.getContext());

    //Value outReplacement =
    //    rewriter
    //        .create<linalgx::Relayout>(
    //            loc, outUnBlockedTensorType, newReluOp->getResult(0),
    //            outUnBlockedTensor, maps.first, maps.second)
    //        .getResult()[0];
    //rewriter.replaceOp(linalgOp, outReplacement);
    //return success();
    return failure();
  }
};
#endif

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

} // end namespace

void mlir::tpp::populateSinkRelayoutPatterns(RewritePatternSet &patterns) {
  // clang-format off
  mlir::tpp::populateMapLinalgToTppPatterns(patterns);
  // patterns.add<SinkBlockLayoutAfterRelu>(patterns.getContext());
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
