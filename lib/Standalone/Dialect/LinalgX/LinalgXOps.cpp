//===- LinalgXOps.cpp - LinalgX dialect ops ---------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/LinalgX/LinalgXOps.h"
#include "Standalone/Dialect/LinalgX/LinalgXDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/MathExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::linalgx;
using namespace mlir::bufferization;

//===----------------------------------------------------------------------===//
// PackOp and UnPackOp canonicalizer
//===----------------------------------------------------------------------===//

LogicalResult PackOp::canonicalize(PackOp packOp, PatternRewriter &rewriter) {
  linalgx::UnPackOp unpackOp =
      packOp.getInput().getDefiningOp<linalgx::UnPackOp>();
  if (!unpackOp)
    return failure();
  rewriter.replaceOp(packOp, unpackOp.getInput());
  return success();
}

LogicalResult UnPackOp::canonicalize(UnPackOp unpackOp,
                                     PatternRewriter &rewriter) {
  return failure();
}

//===----------------------------------------------------------------------===//
// PackOp and UnPackOp builders
//===----------------------------------------------------------------------===//

void PackOp::build(OpBuilder &builder, OperationState &result, Value input,
                   Value output, ArrayAttr outerDimPerm, ArrayAttr innerDimsPos,
                   ArrayRef<OpFoldResult> tiles, Value padding) {
  SmallVector<Value> innerTiles;
  SmallVector<int64_t> staticInnerTiles;
  dispatchIndexOpFoldResults(tiles, innerTiles, staticInnerTiles,
                             ShapedType::kDynamicSize);
  if (!outerDimPerm.size())
    build(builder, result, llvm::None, input, output, /*outerDimPerm=*/{},
          innerDimsPos, innerTiles, builder.getI64ArrayAttr(staticInnerTiles),
          padding);
  else
    build(builder, result, llvm::None, input, output, outerDimPerm,
          innerDimsPos, innerTiles, builder.getI64ArrayAttr(staticInnerTiles),
          padding);
}

void PackOp::build(OpBuilder &builder, OperationState &result, Value input,
                   Value output, ArrayRef<int64_t> innerDimPos,
                   ArrayRef<int64_t> outerDimPerm,
                   ArrayRef<OpFoldResult> tiles) {
  assert(!innerDimPos.empty() && "expect innerDimPos to be non empty");
  assert(!tiles.empty() && "expect tiles to be non empty");
  SmallVector<Value> innerTiles;
  SmallVector<int64_t> staticInnerTiles;
  dispatchIndexOpFoldResults(tiles, innerTiles, staticInnerTiles,
                             ShapedType::kDynamicSize);
  Type typeOutput = output.getType();
  if (outerDimPerm.empty())
    build(builder, result, typeOutput, input, output,
          /*outerDimPerm=*/{}, builder.getI64ArrayAttr(innerDimPos), innerTiles,
          builder.getI64ArrayAttr(staticInnerTiles), /*padding_value=*/{});
  else
    build(builder, result, typeOutput, input, output,
          builder.getI64ArrayAttr(outerDimPerm),
          builder.getI64ArrayAttr(innerDimPos), innerTiles,
          builder.getI64ArrayAttr(staticInnerTiles), /*padding_value=*/{});
}

void UnPackOp::build(OpBuilder &builder, OperationState &result, Value input,
                     Value output, ArrayRef<int64_t> dimPos,
                     ArrayRef<OpFoldResult> tiles) {
  SmallVector<Value> innerTiles;
  SmallVector<int64_t> staticInnerTiles;
  dispatchIndexOpFoldResults(tiles, innerTiles, staticInnerTiles,
                             ShapedType::kDynamicSize);
  Type typeOutput = output.getType();
  build(builder, result, typeOutput, input, output, /*output_dims_pos=*/{},
        builder.getI64ArrayAttr(dimPos), innerTiles,
        builder.getI64ArrayAttr(staticInnerTiles));
}

void UnPackOp::build(OpBuilder &builder, OperationState &result, Value input,
                     Value output, ArrayAttr outerDimPerm,
                     ArrayAttr innerDimsPos, ArrayRef<OpFoldResult> tiles) {
  SmallVector<Value> innerTiles;
  SmallVector<int64_t> staticInnerTiles;
  dispatchIndexOpFoldResults(tiles, innerTiles, staticInnerTiles,
                             ShapedType::kDynamicSize);
  if (!outerDimPerm.size())
    build(builder, result, llvm::None, input, output, /*outerDimPerm=*/{},
          innerDimsPos, innerTiles, builder.getI64ArrayAttr(staticInnerTiles));
  else
    build(builder, result, llvm::None, input, output, outerDimPerm,
          innerDimsPos, innerTiles, builder.getI64ArrayAttr(staticInnerTiles));
}

//===----------------------------------------------------------------------===//
// PackOp and UnPackOp utils
//===----------------------------------------------------------------------===//

static void getEffectsImpl(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects,
    ValueRange results, ValueRange inputBuffers, ValueRange outputBuffers) {
  for (Value value : results) {
    effects.emplace_back(MemoryEffects::Allocate::get(), value,
                         SideEffects::DefaultResource::get());
  }
  for (Value value : inputBuffers) {
    effects.emplace_back(MemoryEffects::Read::get(), value,
                         SideEffects::DefaultResource::get());
  }
  for (Value value : outputBuffers) {
    effects.emplace_back(MemoryEffects::Read::get(), value,
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), value,
                         SideEffects::DefaultResource::get());
  }
}

static Value getDimValue(OpBuilder &builder, Location loc, Value v,
                         int64_t dim) {
  return TypeSwitch<Type, Value>(v.getType())
      .Case<RankedTensorType>([&](RankedTensorType t) -> Value {
        return builder.create<tensor::DimOp>(loc, v, dim);
      })
      .Case<MemRefType>([&](MemRefType t) -> Value {
        return builder.create<memref::DimOp>(loc, v, dim);
      })
      .Default([&](Type t) { return Value(); });
}

static OpFoldResult getDim(OpBuilder &builder, Location loc, Value v,
                           int64_t dim) {
  auto t = v.getType().cast<ShapedType>();
  if (t.isDynamicDim(dim)) {
    return getDimValue(builder, loc, v, dim);
  }
  return builder.getI64IntegerAttr(t.getDimSize(dim));
}

/// Return true if at least one element in `tiles` is zero.
static bool hasZeros(ArrayRef<OpFoldResult> tiles) {
  return llvm::any_of(
      tiles, [&](OpFoldResult tile) { return isConstantIntValue(tile, 0); });
}

/// Return true if `dimsPos` is invalid. It is invalid when: a) it contains
/// duplicate. b) At least one dimension is out of bound (`dimPos` is >= 0 and <
/// rank). c) the number of elements in `dimsPos` is > thank `rank`.
static bool isInvalid(ArrayRef<int64_t> dimsPos, int64_t rank) {
  int64_t size = dimsPos.size();
  if (size > rank)
    return true;
  DenseSet<int64_t> uniqued;
  for (int64_t dim : dimsPos)
    uniqued.insert(dim);
  if (dimsPos.size() != uniqued.size())
    return true;
  return llvm::any_of(
      dimsPos, [rank](int64_t dimPos) { return dimPos < 0 || dimPos >= rank; });
}

/// Check if we have enough static information to catch undefined behavior when
/// the tile size does not divide perfectly the dimension of the input tensor.
static bool areNotFullTiles(ArrayRef<int64_t> inputShape,
                            DenseMap<int64_t, OpFoldResult> dimAndTileMapping) {
  int64_t rank = inputShape.size();
  for (int64_t dim = 0; dim < rank; dim++) {
    if (inputShape[dim] == ShapedType::kDynamicSize)
      continue;
    if (dimAndTileMapping.count(dim)) {
      Optional<int64_t> constantTile =
          getConstantIntValue(dimAndTileMapping[dim]);
      if (!constantTile)
        continue;
      if (inputShape[dim] % (*constantTile) != 0)
        return true;
    }
  }
  return false;
}

/// Check if two `RankedShapedTypes` are compatible. The shapes are compatible
/// if there are no statically known shapes that mismatch. Shapes are still
/// compatible if one is static and other is dynamic.
static bool isCompatible(ShapedType a, ShapedType b) {
  if (a.getRank() != b.getRank())
    return false;
  for (auto it : llvm::zip(a.getShape(), b.getShape())) {
    auto aDim = std::get<0>(it);
    auto bDim = std::get<1>(it);
    if (!ShapedType::isDynamic(aDim) && !ShapedType::isDynamic(bDim) &&
        aDim != bDim)
      return false;
  }
  return true;
}

/// Interchange `elements` starting at offset `offset` based on the indexes in
/// `interchangeVector`.
template <typename T>
static SmallVector<T> interchange(ArrayRef<T> elements,
                                  ArrayRef<int64_t> interchangeVector,
                                  int64_t offset) {
  SmallVector<T> rearrangedElements = llvm::to_vector(elements);
  if (interchangeVector.empty())
    return rearrangedElements;
  for (auto en : llvm::enumerate(interchangeVector)) {
    rearrangedElements[en.index() + offset] = elements[en.value() + offset];
  }
  return rearrangedElements;
}

/// Return the `interchangeVector` based on `dims_pos`.
static SmallVector<int64_t>
computeInterchangeFromDimPos(ArrayRef<int64_t> innerDimsPos,
                             int64_t inputRank) {
  SmallVector<int64_t> interchangeVector;
  interchangeVector.reserve(innerDimsPos.size());
  // First map dims and their position. For example, dims_pos = [2, 0] will map
  // to:
  // [
  //  [ key: 2, value: 0]
  //  [ key: 0, value: 1]
  // ]
  // where key is the idx in dims_pos while value its position in dims_pos.
  DenseMap<int64_t, int64_t> dimsAndPosMapping;
  for (int64_t dimsIdx = 0, end = innerDimsPos.size(); dimsIdx < end; dimsIdx++)
    dimsAndPosMapping[innerDimsPos[dimsIdx]] = dimsIdx;

  // Scan the position in order and insert the value in the map
  // to compute the interchange vector.
  for (int64_t dimsIdx = 0; dimsIdx < inputRank; dimsIdx++) {
    if (dimsAndPosMapping.count(dimsIdx))
      interchangeVector.push_back(dimsAndPosMapping[dimsIdx]);
  }
  return interchangeVector;
}

/// Utility function shared between Pack and UnPack to get the tile sizes as
/// OpFoldResults.
// TODO: interface or base class in .td
template <typename OpTy>
static SmallVector<OpFoldResult> getMixedTiles(OpTy op) {
  static_assert(llvm::is_one_of<OpTy, PackOp, UnPackOp>::value,
                "applies to only pack or unpack operations");
  SmallVector<OpFoldResult> mixedInnerTiles;
  unsigned dynamicValIndex = 0;
  for (Attribute attr : op.getStaticInnerTiles()) {
    auto tileAttr = attr.cast<IntegerAttr>();
    if (!ShapedType::isDynamic(tileAttr.getInt()))
      mixedInnerTiles.push_back(tileAttr);
    else
      mixedInnerTiles.push_back(op.getInnerTiles()[dynamicValIndex++]);
  }
  return mixedInnerTiles;
}

/// Return the tile sizes as `int64_t`. If a tile size is dynamic a sentinel
/// `kDynamicSize` is introduced at that position in the returned vector.
template <typename OpTy> static SmallVector<int64_t> getStaticTiles(OpTy op) {
  static_assert(llvm::is_one_of<OpTy, PackOp, UnPackOp>::value,
                "applies to only pack or unpack operations");
  SmallVector<Value> dynamicTiles;
  SmallVector<int64_t> staticTiles;
  dispatchIndexOpFoldResults(op.getMixedTiles(), dynamicTiles, staticTiles,
                             ShapedType::kDynamicSize);
  return staticTiles;
}

/// Utility function shared between Pack and UnPack to get a map between
/// `dim_pos` and `inner_tiles`.
// TODO: interface or base class in .td
template <typename OpTy>
static DenseMap<int64_t, OpFoldResult> getDimAndTileMapping(OpTy op) {
  static_assert(llvm::is_one_of<OpTy, PackOp, UnPackOp>::value,
                "applies to only pack or unpack operations");
  DenseMap<int64_t, OpFoldResult> dimAndTileMapping;
  SmallVector<int64_t> dimsToBlock =
      extractFromI64ArrayAttr(op.getInnerDimsPos());
  SmallVector<OpFoldResult> tiles = op.getMixedTiles();
  assert(tiles.size() == dimsToBlock.size() &&
         "tiles must match indices of dimension to block");
  // bind the dimension with the tile factor.
  for (auto i : llvm::seq<int64_t>(0, dimsToBlock.size()))
    dimAndTileMapping[dimsToBlock[i]] = tiles[i];
  return dimAndTileMapping;
}

/// Utility fuction to build the iteration domain for `packOp` or `unPackOp`.
template <typename OpTy>
static SmallVector<Range> getIterationDomain(OpTy op, OpBuilder &builder) {
  static_assert(llvm::is_one_of<OpTy, PackOp, UnPackOp>::value,
                "applies to only pack or unpack operations");
  OpBuilder::InsertionGuard g(builder);
  Location loc = op.getLoc();
  int64_t rank = (std::is_same<OpTy, PackOp>::value) ? op.getInputRank()
                                                     : op.getOutputRank();
  SmallVector<Range> loopBounds(rank);
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  ReifiedRankedShapedTypeDims resultShape;
  (void)op.reifyResultShapes(builder, resultShape);
  for (auto dim : llvm::seq<int64_t>(0, rank)) {
    loopBounds[dim].offset = zero;
    loopBounds[dim].stride = one;
    loopBounds[dim].size = resultShape[0][dim];
  }
  return loopBounds;
}

/// Common verifier for `PackOp` and `UnPackOp`.
template <typename OpTy>
static LogicalResult commonVerifierPackAndUnPackOp(OpTy packOrUnPack) {
  static_assert(llvm::is_one_of<OpTy, PackOp, UnPackOp>::value,
                "applies to only pack or unpack operations");
  Operation *op = packOrUnPack.getOperation();
  int64_t rank = (std::is_same<OpTy, PackOp>::value)
                     ? packOrUnPack.getInputRank()
                     : packOrUnPack.getOutputRank();
  SmallVector<int64_t> innerDimsPos =
      extractFromI64ArrayAttr(packOrUnPack.getInnerDimsPos());
  SmallVector<int64_t> outerDimsPos =
      extractFromI64ArrayAttr(packOrUnPack.getOuterDimsPerm());
  // Verify tiles. Make sure each provided tile is non-zero.
  if (hasZeros(packOrUnPack.getMixedTiles()))
    return op->emitError("invalid tile factor");
  if (isInvalid(innerDimsPos, rank))
    return op->emitError("invalid inner_dims_pos vector");
  if (isInvalid(outerDimsPos, rank))
    return op->emitError("invalid outer_dims_pos vector");
  if (packOrUnPack.getMixedTiles().size() != innerDimsPos.size()) {
    return op->emitError(
        "blocking factors must equal the number of dimensions to block");
  }
  return success();
}

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

//===----------------------------------------------------------------------===//
// PackOp
//===----------------------------------------------------------------------===//

/// verifier for the pack operation.
LogicalResult PackOp::verify() {
  Operation *op = getOperation();
  size_t numberOfBlockingFactors = getMixedTiles().size();
  SmallVector<int64_t> innerDimsPos =
      extractFromI64ArrayAttr(getInnerDimsPos());
  if (failed(commonVerifierPackAndUnPackOp(*this))) {
    return failure();
  }

  // Blocking factors must be less or equal than the input rank, and must
  // match the number of `dims_pos`.
  if (numberOfBlockingFactors > getInputRank()) {
    return op->emitError(
        "blocking factors must be less or equal than the input rank");
  }

  // Require output rank to match input rank + number of blocking factors.
  if ((getInputRank() + numberOfBlockingFactors) != getOutputRank()) {
    return op->emitError(
        "output rank must equal input rank + blocking factors");
  }

  // Bail out if the tile does not divide the dimension fully. In the case of
  // dynamic tile factors or dimensions, having a partial tile is undefined
  // behavior.
  if (!getPaddingValue() &&
      areNotFullTiles(getInputShape(), getDimAndTileMapping())) {
    return op->emitError("invalid tile factor provided. Only full tiles are "
                         "supported when padding_value is not set");
  }
  // Verify result type against inferred type.
  SmallVector<int64_t> outerDimsPos =
      extractFromI64ArrayAttr(getOuterDimsPerm());
  DenseMap<int64_t, OpFoldResult> tileAndPosMapping = getDimAndTileMapping();
  ShapedType expectedOutputType = inferPackedType(
      getInputType(), getStaticTiles(), tileAndPosMapping, outerDimsPos);
  if (!isCompatible(expectedOutputType, getOutputType())) {
    return op->emitError(
               "infered type do not match provided output type. Expected ")
           << expectedOutputType << " but got: " << getOutputType();
  }

  if (auto paddingValue = getPaddingValue()) {
    if (paddingValue.getType() != expectedOutputType.getElementType()) {
      return op->emitError("expected padding_value has ")
             << expectedOutputType.getElementType()
             << " but got: " << paddingValue.getType();
    }
  }
  return success();
}

/// Get the tile sizes as `OpFoldResult`.
SmallVector<OpFoldResult> PackOp::getMixedTiles() {
  return ::getMixedTiles(*this);
}

SmallVector<int64_t> PackOp::getStaticTiles() {
  return ::getStaticTiles(*this);
}

SmallVector<utils::IteratorType> PackOp::getLoopIteratorTypes() {
  // Note that here we consider only the tiled loops, the point loops are
  // materialized when building the body of the operation.
  SmallVector<utils::IteratorType> iteratorTypes(getInputRank(),
                                                 utils::IteratorType::parallel);
  return iteratorTypes;
}

DenseMap<int64_t, OpFoldResult> PackOp::getDimAndTileMapping() {
  return ::getDimAndTileMapping(*this);
}

SmallVector<Range> PackOp::getIterationDomain(OpBuilder &builder) {
  return ::getIterationDomain(*this, builder);
}
/// Generate the body of the innermost loop of the scalar implementation
/// of `pack` operation.
static void generatePackOpScalarImplementationBody(PackOp packOp,
                                                   OpBuilder &builder,
                                                   Location loc,
                                                   ValueRange ivs) {
  // Note: `ivs` are already in the correct order, possibly interchanged based
  // on `dims_pos`. However, connecting the loops with the access patterns is
  // difficult - What is the relation between the position of the tile loop and
  // the point loop? However, if we interchange `ivs` once more to go to the
  // canonical blocking format: ABCabc, this connection becomes trivial: Each
  // point loop is pointLoopsOffset + inputRank away from the tiled loop.
  SmallVector<int64_t> dimsToInnerBlock =
      extractFromI64ArrayAttr(packOp.getInnerDimsPos());
  SmallVector<int64_t> dimsToOuterBlock =
      extractFromI64ArrayAttr(packOp.getOuterDimsPerm());

  SmallVector<Value> interchangedIvs = ivs;
  SmallVector<int64_t> interchangeVector =
      computeInterchangeFromDimPos(dimsToInnerBlock, packOp.getInputRank());
  interchangedIvs = interchange<Value>(interchangedIvs, interchangeVector,
                                       /*offset=*/packOp.getInputRank());
  if (!dimsToOuterBlock.empty()) {
    interchangeVector =
        computeInterchangeFromDimPos(dimsToOuterBlock, packOp.getInputRank());
    interchangedIvs =
        interchange<Value>(interchangedIvs, interchangeVector, /*offset=*/0);
  }

  SmallVector<OpFoldResult> tiles = packOp.getMixedTiles();
  DenseMap<int64_t, OpFoldResult> dimAndTileMapping =
      packOp.getDimAndTileMapping();
  SmallVector<OpFoldResult> sourceIndices;
  size_t pointLoopsOffset = 0;
  int64_t inputRank = packOp.getInputRank();
  for (auto dim : llvm::seq<int64_t>(0, inputRank)) {
    if (dimAndTileMapping.count(dim)) {
      AffineExpr i, j, tile;
      bindDims(builder.getContext(), i, j);
      bindSymbols(builder.getContext(), tile);
      OpFoldResult sourceIndex = makeComposedFoldedAffineApply(
          builder, loc, i * tile + j,
          ArrayRef<OpFoldResult>{
              interchangedIvs[dim],
              interchangedIvs[pointLoopsOffset + packOp.getInputRank()],
              dimAndTileMapping[dim]});
      sourceIndices.push_back(sourceIndex);
      ++pointLoopsOffset;
    } else {
      sourceIndices.push_back(interchangedIvs[dim]);
    }
  }

  auto createLoad = [&]() -> Value {
    return builder.create<memref::LoadOp>(
        loc, packOp.getInput(), getAsValues(builder, loc, sourceIndices));
  };
  Value scalar;
  if (auto paddingValue = packOp.getPaddingValue()) {
    ArithBuilder arithBuilder(builder, loc);
    Value isInBounds;
    for (auto dim : llvm::seq<int64_t>(0, inputRank)) {
      Value idx =
          getValueOrCreateConstantIndexOp(builder, loc, sourceIndices[dim]);
      Value cond = arithBuilder.slt(
          idx, getDimValue(builder, loc, packOp.getInput(), dim));
      isInBounds = dim == 0 ? cond : arithBuilder._and(isInBounds, cond);
    }
    scalar = builder
                 .create<scf::IfOp>(
                     loc, packOp.getElementType(), isInBounds, /*thenBuilder=*/
                     [&](OpBuilder &b, Location l) {
                       b.create<scf::YieldOp>(l, createLoad());
                     },
                     /*elseBuilder=*/
                     [&](OpBuilder &b, Location l) {
                       b.create<scf::YieldOp>(l, paddingValue);
                     })
                 .getResult(0);
  } else {
    scalar = createLoad();
  }

  builder.create<memref::StoreOp>(loc, scalar, packOp.getOutput(), ivs);
}

LogicalResult PackOp::generateScalarImplementation(OpBuilder &builder,
                                                   Location loc,
                                                   ValueRange ivs) {
  OpBuilder::InsertionGuard g(builder);
  // The `ivs` already represent the position into the output tensor for the
  // non data-tile dimensions.
  SmallVector<Value> ivVec = llvm::to_vector(ivs);
  ReifiedRankedShapedTypeDims outputShape;
  if (failed(reifyResultShapes(builder, outputShape)))
    return getOperation()->emitOpError("failed to reify result shape");
  if (outputShape.size() != 1 || outputShape[0].size() != getOutputRank()) {
    return getOperation()->emitOpError(
               "expected shape of one result value of rank")
           << getOutputRank();
  }

  // Generate the loops that iterate over the data tile.
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);

  // All loops except the innermost are simple loops that just iterate
  // over the tile dimensions.
  for (auto dataTileDim :
       llvm::seq<unsigned>(getInputRank(), getOutputRank() - 1)) {
    Value ub = outputShape[0][dataTileDim];
    scf::ForOp loop = builder.create<scf::ForOp>(loc, zero, ub, one);
    builder.setInsertionPointToStart(loop.getBody());
    ivVec.push_back(loop.getInductionVar());
  }
  // The body of the innermost loops does the actual data movement.
  builder.create<scf::ForOp>(loc, zero, outputShape[0].back(), one,
                             ValueRange{},
                             [&](OpBuilder &bodyBuilder, Location bodyLoc,
                                 Value iv, ValueRange regionIterArgs) {
                               ivVec.push_back(iv);
                               generatePackOpScalarImplementationBody(
                                   *this, bodyBuilder, bodyLoc, ivVec);
                               bodyBuilder.create<scf::YieldOp>(bodyLoc);
                             });
  return success();
}

LogicalResult
PackOp::reifyResultShapes(OpBuilder &builder,
                          ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(getOperation());
  Location loc = getLoc();
  SmallVector<OpFoldResult> mixedTiles = getMixedTiles();
  DenseMap<int64_t, OpFoldResult> dimAndTileMapping = getDimAndTileMapping();

  // Build the output dimension at pos `dimIdx`.
  auto buildOutputDim = [&](OpBuilder &builder, size_t dimIdx) -> OpFoldResult {
    ArrayRef<int64_t> outputShape = getOutputShape();
    if (!ShapedType::isDynamic(outputShape[dimIdx])) {
      return builder.getI64IntegerAttr(outputShape[dimIdx]);
    }

    // Inner tile sizes can be derived from inner_tiles.
    if (dimIdx >= getInputRank()) {
      return mixedTiles[dimIdx - getInputRank()];
    }

    OpFoldResult dimBound = getDim(builder, loc, getInput(), dimIdx);
    if (dimAndTileMapping.count(dimIdx)) {
      AffineExpr dim = builder.getAffineSymbolExpr(0);
      AffineExpr tile = builder.getAffineSymbolExpr(1);
      dimBound = makeComposedFoldedAffineApply(
          builder, loc, dim.ceilDiv(tile),
          ArrayRef<OpFoldResult>{dimBound, dimAndTileMapping[dimIdx]});
    }
    return dimBound;
  };

  reifiedReturnShapes.resize(1);
  reifiedReturnShapes[0].reserve(getOutputRank());
  for (auto dimIdx : llvm::seq<int64_t>(0, getOutputRank())) {
    reifiedReturnShapes[0].push_back(getValueOrCreateConstantIndexOp(
        builder, loc, buildOutputDim(builder, dimIdx)));
  }
  return success();
}

void PackOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  SmallVector<Value> inputBuffers;
  if (getInput().getType().isa<MemRefType>())
    inputBuffers.push_back(getInput());
  SmallVector<Value> outputBuffers;
  if (getOutput().getType().isa<MemRefType>())
    outputBuffers.push_back(getOutput());
  getEffectsImpl(effects, getOperation()->getResults(), inputBuffers,
                 outputBuffers);
}

//===----------------------------------------------------------------------===//
// UnPackOp
//===----------------------------------------------------------------------===//

SmallVector<OpFoldResult> UnPackOp::getMixedTiles() {
  return ::getMixedTiles(*this);
}

SmallVector<int64_t> UnPackOp::getStaticTiles() {
  return ::getStaticTiles(*this);
}

DenseMap<int64_t, OpFoldResult> UnPackOp::getDimAndTileMapping() {
  return ::getDimAndTileMapping(*this);
}

LogicalResult UnPackOp::generateScalarImplementation(OpBuilder &builder,
                                                     Location loc,
                                                     ValueRange ivs) {
  assert(ivs.size() == getOutputRank() &&
         "number of ivs must match the rank of the output tensor");
  OpBuilder::InsertionGuard g(builder);
  ReifiedRankedShapedTypeDims outputShape;
  if (failed(reifyResultShapes(builder, outputShape)))
    return getOperation()->emitOpError("failed to reify result shape");
  if (outputShape.size() != 1 || outputShape[0].size() != getOutputRank()) {
    return getOperation()->emitOpError(
               "expected shape of one result value of rank")
           << getOutputRank();
  }

  DenseMap<int64_t, OpFoldResult> dimAndTileMapping = getDimAndTileMapping();
  // untiled loops and tile loops induction variables.
  SmallVector<Value> inputIvs;
  // point loops induction variables.
  SmallVector<Value> inputIvsPointLoops;
  inputIvs.reserve(getOutputRank());
  inputIvsPointLoops.reserve(dimAndTileMapping.size());
  for (auto dim : llvm::seq<int64_t>(0, getOutputRank())) {
    if (dimAndTileMapping.count(dim)) {
      DivModValue divMod = getDivMod(builder, loc, ivs[dim],
                                     getValueOrCreateConstantIndexOp(
                                         builder, loc, dimAndTileMapping[dim]));
      inputIvsPointLoops.push_back(divMod.remainder);
      inputIvs.push_back(divMod.quotient);
    } else {
      inputIvs.push_back(ivs[dim]);
    }
  }

  // TODO: (lorenzo) simplify the logic a bit. There is `ivs`,
  // `inputIvsPointLoops` and `inputIvs`.
  assert(inputIvsPointLoops.size() + inputIvs.size() == getInputRank() &&
         "expect same number of iduction variables equals to input rank");
  // interchange the point loops induction variables based on `inner_dim_pos`.
  SmallVector<int64_t> innerDims = extractFromI64ArrayAttr(getInnerDimsPos());
  SmallVector<int64_t> interchangeVector =
      computeInterchangeFromDimPos(innerDims, getOutputRank());
  SmallVector<Value> interchangedInputIvsPointLoops = inputIvsPointLoops;
  interchangedInputIvsPointLoops = interchange<Value>(
      interchangedInputIvsPointLoops, interchangeVector, /*offset=*/0);
  // interchange the tiled loops induction variables based on `outer_dims_pos`.
  SmallVector<int64_t> outerDims = extractFromI64ArrayAttr(getOuterDimsPerm());
  if (!outerDims.empty()) {
    interchangeVector =
        computeInterchangeFromDimPos(outerDims, getOutputRank());
    inputIvs = interchange<Value>(inputIvs, interchangeVector, /*offset=*/0);
  }

  llvm::append_range(inputIvs, interchangedInputIvsPointLoops);
  Value scalar = builder.create<memref::LoadOp>(loc, getInput(), inputIvs);
  builder.create<memref::StoreOp>(loc, scalar, getOutput(), ivs);
  return success();
}

template <typename Ty, typename DimOpTy>
static void getDimValues(OpBuilder &b, Location loc, Value v, Ty t,
                         SmallVector<Value> &dimVals) {
  for (auto dim : llvm::enumerate(t.getShape())) {
    if (ShapedType::isDynamic(dim.value())) {
      dimVals.push_back(b.create<DimOpTy>(loc, v, dim.index()));
    } else {
      dimVals.push_back(b.create<arith::ConstantIndexOp>(loc, dim.value()));
    }
  }
}

LogicalResult
UnPackOp::reifyResultShapes(OpBuilder &b,
                            ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  Operation *op = getOperation();
  Value output = getOutput();
  SmallVector<Value> dims;
  Type outputType = output.getType();
  if (auto rankedTensorType = outputType.dyn_cast<RankedTensorType>()) {
    getDimValues<RankedTensorType, tensor::DimOp>(b, op->getLoc(), output,
                                                  rankedTensorType, dims);
  } else if (auto memrefType = outputType.dyn_cast<MemRefType>()) {
    getDimValues<MemRefType, memref::DimOp>(b, op->getLoc(), output, memrefType,
                                            dims);
  } else if (!outputType.isIntOrIndexOrFloat()) {
    return op->emitOpError("invalid type for output operand, expected tensor, "
                           "memref or scalar type");
  }
  reifiedReturnShapes.emplace_back(std::move(dims));
  return success();
}

SmallVector<Range> UnPackOp::getIterationDomain(OpBuilder &builder) {
  return ::getIterationDomain(*this, builder);
}

LogicalResult UnPackOp::verify() {
  Operation *op = getOperation();
  size_t numberOfBlockingFactors = getMixedTiles().size();
  SmallVector<int64_t> innerDimsPos =
      extractFromI64ArrayAttr(getInnerDimsPos());
  if (failed(commonVerifierPackAndUnPackOp(*this))) {
    return failure();
  }

  // Blocking factors must be less or equal than the output rank, and must
  // match the number of `dims_pos`.
  if (numberOfBlockingFactors > getOutputRank()) {
    return op->emitError(
        "blocking factors must be less or equal than the output rank");
  }

  // Require input rank to match output rank + number of blocking factors.
  if ((getOutputRank() + numberOfBlockingFactors) != getInputRank()) {
    return op->emitError(
        "input rank must equal output rank + blocking factors");
  }

  // Verify input type against inferred type. The check includes the cases for
  // incompilete tiles. We allow to `undo` the padding done in the pack.
  SmallVector<int64_t> outerDimsPos =
      extractFromI64ArrayAttr(getOuterDimsPerm());
  DenseMap<int64_t, OpFoldResult> tileAndPosMapping = getDimAndTileMapping();
  ShapedType expectedInputType = inferPackedType(
      getOutputType(), getStaticTiles(), tileAndPosMapping, outerDimsPos);
  if (!isCompatible(expectedInputType, getInputType())) {
    return op->emitError(
               "infered type do not match provided input type. Expected ")
           << expectedInputType << " but got: " << getInputType();
  }
  return success();
}

SmallVector<utils::IteratorType> UnPackOp::getLoopIteratorTypes() {
  SmallVector<utils::IteratorType> iteratorTypes(getOutputRank(),
                                                 utils::IteratorType::parallel);
  return iteratorTypes;
}

void UnPackOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  SmallVector<Value> inputBuffers;
  if (getInput().getType().isa<MemRefType>())
    inputBuffers.push_back(getInput());
  SmallVector<Value> outputBuffers;
  if (getOutput().getType().isa<MemRefType>())
    outputBuffers.push_back(getOutput());
  getEffectsImpl(effects, getOperation()->getResults(), inputBuffers,
                 outputBuffers);
}

#define GET_OP_CLASSES
#include "Standalone/Dialect/LinalgX/LinalgXOps.cpp.inc"
