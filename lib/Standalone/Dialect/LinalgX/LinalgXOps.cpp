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

using RegionBuilderFn = llvm::function_ref<void(ImplicitLocOpBuilder &, Block &,
                                                ArrayRef<NamedAttribute>)>;
// taken from linalgOps.cpp
/// Fills the region of a structured operation using the provided
/// `regionBuilder`. The method is used by both named structured ops created by
/// ods-gen and by manually defined C++ ops. It is called by both builders and
/// parsers and creates a block with arguments corresponding to the elemental
/// types of `inputTypes` and `outputTypes`. All output types are asserted to be
/// ShapedType.
static void fillStructuredOpRegion(OpBuilder &opBuilder, Region &region,
                                   TypeRange inputTypes, TypeRange outputTypes,
                                   ArrayRef<NamedAttribute> attrs,
                                   RegionBuilderFn regionBuilder) {
  assert(llvm::all_of(outputTypes, [](Type t) { return t.isa<ShapedType>(); }));

  // TODO: atm all operands go through getElementTypeOrSelf,
  // reconsider when we have evidence we need to.
  SmallVector<Type, 8> argTypes;
  SmallVector<Location, 8> argLocs;
  for (auto containers : {inputTypes, outputTypes}) {
    for (auto t : containers) {
      argTypes.push_back(getElementTypeOrSelf(t));

      // TODO: Pass in a proper location here.
      argLocs.push_back(opBuilder.getUnknownLoc());
    }
  }

  // RAII.
  OpBuilder::InsertionGuard guard(opBuilder);
  Block *body =
      opBuilder.createBlock(&region, /*insertPt=*/{}, argTypes, argLocs);

  opBuilder.setInsertionPointToStart(body);
  ImplicitLocOpBuilder b(opBuilder.getUnknownLoc(), opBuilder);
  regionBuilder(b, *body, attrs);

  // indexing_maps is an auto-generated method.

  // iterator_types is an auto-generated method.
}

// taken from LinalgOps.cpp
/// Creates a structured operation given `inputs`, `outputs`, `inputMap`,
/// `outputMap` and `attributes`. The result types are derived automatically if
/// `resultTensorTypes` is none. The body of the operation is filled using
/// `regionBuilder`. All ods-gen created structured operations use the method to
/// implement their builders.
static void buildStructuredOp(OpBuilder &b, OperationState &state,
                              llvm::Optional<TypeRange> resultTensorTypes,
                              ValueRange inputs, ValueRange outputs,
                              AffineMap inputMap, AffineMap outputMap,
                              ArrayRef<NamedAttribute> attributes,
                              RegionBuilderFn regionBuilder) {
  // Derive the result types if needed.
  SmallVector<Type> derivedResultTypes =
      resultTensorTypes.value_or(TypeRange());
  if (!resultTensorTypes)
    copy_if(outputs.getTypes(), std::back_inserter(derivedResultTypes),
            [](Type type) { return type.isa<RankedTensorType>(); });

  state.addOperands(inputs);
  state.addOperands(outputs);
  state.addTypes(derivedResultTypes);
  state.addAttributes(attributes);
  // These are our extensions
  state.addAttribute("inputMap", AffineMapAttr::get(inputMap));
  state.addAttribute("outputMap", AffineMapAttr::get(outputMap));
  // end our extensions
  state.addAttribute(
      "operand_segment_sizes",
      b.getI32VectorAttr({static_cast<int32_t>(inputs.size()),
                          static_cast<int32_t>(outputs.size())}));

  // Create and fill the region of the structured operation.
  Region &region = *state.addRegion();
  fillStructuredOpRegion(b, region, TypeRange(inputs), TypeRange(outputs),
                         state.attributes.getAttrs(), regionBuilder);
}

//===----------------------------------------------------------------------===//
// Relayout
//===----------------------------------------------------------------------===//

ArrayAttr Relayout::getIndexingMaps() {
  MLIRContext *context = getContext();
  auto maybeInputMap = getInputMap();
  auto maybeOutputMap = getOutputMap();
  int64_t inputRank = getRank(getInputOperand(0));
  int64_t outputRank = getRank(getOutputOperand(0));
  return Builder(getContext())
      .getAffineMapArrayAttr(
          {linalg::extractOrIdentityMap(maybeInputMap, inputRank, context),
           linalg::extractOrIdentityMap(maybeOutputMap, outputRank, context)});
}

ArrayAttr Relayout::iterator_types() {
  int64_t numLoops = getMatchingIndexingMap(getInputOperand(0)).getNumDims();
  return Builder(getContext())
      .getStrArrayAttr(
          SmallVector<StringRef, 8>(numLoops, getParallelIteratorTypeName()));
}

std::string Relayout::getLibraryCallName() {
  return "relayout_to_block_layout_and_back";
}

void Relayout::regionBuilder(ImplicitLocOpBuilder &b, Block &block,
                             llvm::ArrayRef<NamedAttribute> attrs = {}) {
  assert(block.getNumArguments() == 2 && "CopyOp regionBuilder expects 2 args");
  b.create<linalg::YieldOp>(block.getArgument(0));
}

static void getGenericEffectsImpl(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects,
    ValueRange results, ValueRange inputBuffers, ValueRange outputs) {
  for (Value value : inputBuffers)
    effects.emplace_back(MemoryEffects::Read::get(), value,
                         SideEffects::DefaultResource::get());

  for (Value value : outputs)
    effects.emplace_back(MemoryEffects::Write::get(), value,
                         SideEffects::DefaultResource::get());
}

void Relayout::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  SmallVector<Value> inputBuffers = getInputBufferOperands();
  SmallVector<Value> outputBuffers = getOutputBufferOperands();
  getGenericEffectsImpl(effects, getOperation()->getResults(), inputBuffers,
                        outputBuffers);
}

static ParseResult
parseInputOutputAndMaps(OpAsmParser &parser, OperationState &result,
                        Type &inputType, AffineMapAttr &inputMap,
                        Type &outputType, AffineMapAttr &outputMap) {
  OpAsmParser::UnresolvedOperand inputOperand, outputOperand;
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (succeeded(parser.parseKeyword("ins")))
    if (parser.parseLParen())
      return failure();

  if (parser.parseOperand(inputOperand) || parser.parseColonType(inputType) ||
      parser.parseComma())
    return failure();

  if (parser.parseCustomAttributeWithFallback(inputMap, Type(), "inputMap",
                                              result.attributes) ||
      parser.parseRParen())
    return failure();

  if (succeeded(parser.parseKeyword("outs")))
    if (parser.parseLParen())
      return failure();

  if (parser.parseOperand(outputOperand) || parser.parseColonType(outputType) ||
      parser.parseComma())
    return failure();

  if (parser.parseCustomAttributeWithFallback(outputMap, Type(), "outputMap",
                                              result.attributes) ||
      parser.parseRParen())
    return failure();

  if (parser.resolveOperand(inputOperand, inputType, result.operands) ||
      parser.resolveOperand(outputOperand, outputType, result.operands))
    return failure();

  result.addAttribute("operand_segment_sizes",
                      parser.getBuilder().getI32VectorAttr({1, 1}));

  return success();
}

ParseResult Relayout::parse(OpAsmParser &parser, OperationState &state) {
  Type inputType, outputType;
  AffineMapAttr inputMap, outputMap;
  if (parseInputOutputAndMaps(parser, state, inputType, inputMap, outputType,
                              outputMap))
    return failure();

  if (parser.parseOptionalArrowTypeList(state.types))
    return failure();

  std::unique_ptr<Region> region = std::make_unique<Region>();
  OpBuilder opBuilder(parser.getContext());
  fillStructuredOpRegion(opBuilder, *region, TypeRange(inputType),
                         TypeRange(outputType), state.attributes.getAttrs(),
                         regionBuilder);
  state.addRegion(std::move(region));
  return success();
}

static void printResult(OpAsmPrinter &printer, TypeRange resultTypes) {
  if (!resultTypes.empty())
    return printer.printOptionalArrowTypeList(resultTypes);
}

static void printOperands(OpAsmPrinter &printer, ValueRange inputs,
                          ValueRange outputs, AffineMap inputMap,
                          AffineMap outputMap) {
  assert(inputs.size() == 1 && "expect single input");
  assert(outputs.size() == 1 && "expect single output");
  Value input = inputs[0];
  Value output = outputs[0];
  printer << " ins(" << input << " : " << input.getType() << ", "
          << AffineMapAttr::get(inputMap) << ")";
  printer << " outs(" << output << " : " << output.getType() << ", "
          << AffineMapAttr::get(outputMap) << ")";
}

void Relayout::print(OpAsmPrinter &printer) {
  // print operands.
  printOperands(printer, getInputs(), getOutputs(), getInputMap(),
                getOutputMap());
  // print results.
  printResult(printer, this->getResultTypes());
  // Region is elided.
}

struct RelayoutOfRelayout : public OpRewritePattern<Relayout> {
  using OpRewritePattern<Relayout>::OpRewritePattern;

  bool canFoldIntoProducer(Relayout producer, Relayout consumer) const {
    Value inputProducer = producer.getOperand(0);
    Value consumerOutput = consumer.getOperand(1);
    return (inputProducer.getType() == consumerOutput.getType()) &&
           (producer.getInputMap() == consumer.getOutputMap());
  }

  LogicalResult matchAndRewrite(Relayout relayout,
                                PatternRewriter &rewriter) const final {
    Relayout producer = relayout.getOperand(0).getDefiningOp<Relayout>();
    if (!producer || !canFoldIntoProducer(producer, relayout))
      return failure();
    rewriter.replaceOp(relayout, producer.getOperand(0));
    return success();
  }
};

void Relayout::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add<RelayoutOfRelayout>(context);
}

//===----------------------------------------------------------------------===//
// PackOp and UnPackOp utils
//===----------------------------------------------------------------------===//

/// Return true if at least one element in `tiles` is zero.
static bool hasZeros(ArrayRef<OpFoldResult> tiles) {
  return llvm::any_of(
      tiles, [&](OpFoldResult tile) { return isConstantIntValue(tile, 0); });
}

/// Return true if `dimsPos` is invalid. It is invalid when: a) it contains
/// duplicate.
static bool isInvalid(ArrayRef<int64_t> dimsPos) {
  DenseSet<int64_t> uniqued;
  for (int64_t dim : dimsPos)
    uniqued.insert(dim);
  return dimsPos.size() != uniqued.size();
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

/// Return true if each element in `dimsPos` is >= 0 and < rank.
static bool isInBound(ArrayRef<int64_t> dimsPos, int64_t rank) {
  return llvm::all_of(
      dimsPos, [rank](int64_t dimPos) { return dimPos >= 0 && dimPos < rank; });
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
  assert((rearrangedElements.size() - offset) == interchangeVector.size() &&
         "number of elements must equal number of permutations");
  for (int64_t idx = 0, end = interchangeVector.size(); idx < end; idx++) {
    rearrangedElements[interchangeVector[idx] + offset] =
        elements[idx + offset];
  }
  return rearrangedElements;
}

/// Return the `interchangeVector` based on `dims_pos`.
static SmallVector<int64_t>
computeInterchangeFromDimPos(ArrayRef<int64_t> dimsPos, int64_t inputRank) {
  SmallVector<int64_t> interchangeVector;
  interchangeVector.reserve(dimsPos.size());
  // First map dims and their position. For example, dims_pos = [2, 0] will map
  // to:
  // [
  //  [ key: 2, value: 0]
  //  [ key: 0, value: 1]
  // ]
  // where key is the idx in dims_pos while value its position in dims_pos.
  DenseMap<int64_t, int64_t> dimsAndPosMapping;
  for (int64_t dimsIdx = 0, end = dimsPos.size(); dimsIdx < end; dimsIdx++)
    dimsAndPosMapping[dimsPos[dimsIdx]] = dimsIdx;

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

/// Utility function shared between Pack and UnPack to get a map between
/// `dim_pos` and `inner_tiles`.
// TODO: interface or base class in .td
template <typename OpTy>
static DenseMap<int64_t, OpFoldResult> getDimAndTileMapping(OpTy op) {
  static_assert(llvm::is_one_of<OpTy, PackOp, UnPackOp>::value,
                "applies to only pack or unpack operations");
  DenseMap<int64_t, OpFoldResult> dimAndTileMapping;
  SmallVector<int64_t> dimsToBlock = extractFromI64ArrayAttr(op.getDimsPos());
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
  SmallVector<int64_t> dimsPos =
      extractFromI64ArrayAttr(packOrUnPack.getDimsPos());
  // Verify tiles. Make sure each provided tile is non-zero.
  if (hasZeros(packOrUnPack.getMixedTiles()))
    return op->emitError("invalid tile factor");
  // Reject `dims_pos` if it contains duplicate.
  if (isInvalid(dimsPos))
    return op->emitError("invalid dims_pos vector");
  if (packOrUnPack.getMixedTiles().size() != dimsPos.size()) {
    return op->emitError(
        "blocking factors must equal the number of dimensions to block");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Pack
//===----------------------------------------------------------------------===//

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

/// Infer result/output type given the input and the tile sizes.
ShapedType PackOp::inferResultType() {
  DenseMap<int64_t, OpFoldResult> tileAndPosMapping = getDimAndTileMapping();
  SmallVector<int64_t> inferredShape;
  inferredShape.reserve(getOutputRank());
  ShapedType inputType = getInputType();
  int64_t rank = getInputRank();

  // tile loop.
  for (auto i : llvm::seq<int64_t>(0, rank)) {
    if (tileAndPosMapping.count(i)) {
      Optional<int64_t> tileSize =
          getConstantIntValue(tileAndPosMapping.lookup(i));
      if (inputType.isDynamicDim(i) || !tileSize) {
        inferredShape.push_back(ShapedType::kDynamicSize);
      } else {
        int64_t sizeTiledDim = ceilDiv(inputType.getDimSize(i), *tileSize);
        inferredShape.push_back(sizeTiledDim);
      }
    } else {
      inferredShape.push_back(inputType.getShape()[i]);
    }
  }

  // point loop.
  auto staticTiles = getStaticTiles();
  inferredShape.append(staticTiles.begin(), staticTiles.end());

  return TypeSwitch<Type, ShapedType>(inputType)
      .Case<RankedTensorType>([&](RankedTensorType t) -> ShapedType {
        return RankedTensorType::get(inferredShape, inputType.getElementType());
      })
      .Case<MemRefType>([&](MemRefType t) -> ShapedType {
        return MemRefType::get(inferredShape, inputType.getElementType());
      })
      .Default([&](Type t) {
        llvm_unreachable("unexpected type");
        return nullptr;
      });
}

/// verifier for the pack operation.
LogicalResult PackOp::verify() {
  Operation *op = getOperation();
  size_t numberOfBlockingFactors = getMixedTiles().size();
  SmallVector<int64_t> dimsPos = extractFromI64ArrayAttr(getDimsPos());
  if (failed(commonVerifierPackAndUnPackOp(*this))) {
    return failure();
  }
  // Blocking factors must be less or equal than the input rank, and must
  // match the number of `dims_pos`.
  if (numberOfBlockingFactors > getInputRank()) {
    return op->emitError(
        "blocking factors must be less or equal than the input rank");
  }
  // Require `dim_pos` to be in-bound. `dim_pos` carries the index of the
  // dimensions to block.
  if (!isInBound(dimsPos, getInputRank()))
    return op->emitError("out-of-bound position");
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
  ShapedType expectedType = inferResultType();
  if (!isCompatible(expectedType, getOutputType())) {
    return op->emitError(
               "infered type do not match provided output type. Expected ")
           << expectedType << " but got: " << getOutputType();
  }

  if (auto paddingValue = getPaddingValue()) {
    if (paddingValue.getType() != expectedType.getElementType()) {
      return op->emitError("expected padding_value has ")
             << expectedType.getElementType()
             << " but got: " << paddingValue.getType();
    }
  }
  return success();
}

/// Get the tile sizes as `OpFoldResult`.
SmallVector<OpFoldResult> PackOp::getMixedTiles() {
  return ::getMixedTiles(*this);
}

/// Return the tile sizes as `int64_t`. If a tile size is dynamic a sentinel
/// `kDynamicSize` is introduced at that position in the returned vector.
SmallVector<int64_t> PackOp::getStaticTiles() {
  SmallVector<Value> dynamicTiles;
  SmallVector<int64_t> staticTiles;
  dispatchIndexOpFoldResults(getMixedTiles(), dynamicTiles, staticTiles,
                             ShapedType::kDynamicSize);
  return staticTiles;
}

// Implement the tiling interface. The number of loops equals
// the rank of the output tensors. All the loops are parallel.
SmallVector<utils::IteratorType> PackOp::getLoopIteratorTypes() {
  SmallVector<utils::IteratorType> iteratorTypes(getInputRank(),
                                                 utils::IteratorType::parallel);
  return iteratorTypes;
}

/// Return a mapping from positions `dims_pos` to their `OpFoldResult` tile
/// factors.
DenseMap<int64_t, OpFoldResult> PackOp::getDimAndTileMapping() {
  return ::getDimAndTileMapping(*this);
}

/// Implements `getIterationDomain` from the tiling interface. In each
/// loop the lower bound is zero and the step is one. For upper bound
/// is inferred from the output tensor for the dimensions that are
/// not part of the data tile created.
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
  SmallVector<int64_t> dimsToBlock =
      extractFromI64ArrayAttr(packOp.getDimsPos());
  SmallVector<Value> interchangedIvs = ivs;
  SmallVector<int64_t> interchangeVector =
      computeInterchangeFromDimPos(dimsToBlock, packOp.getInputRank());
  interchangedIvs = interchange<Value>(interchangedIvs, interchangeVector,
                                       /*offset=*/packOp.getInputRank());

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

/// Implements `generateScalarImplementation` from the tiling interface.
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

//===----------------------------------------------------------------------===//
// UnPack
//===----------------------------------------------------------------------===//

SmallVector<OpFoldResult> UnPackOp::getMixedTiles() {
  return ::getMixedTiles(*this);
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

  assert(inputIvsPointLoops.size() + inputIvs.size() == getInputRank() &&
         "expect same number of iduction variables equals to input rank");
  // interchange the point loops induction variables based on `dim_pos`.
  SmallVector<int64_t> dimsToBlock = extractFromI64ArrayAttr(getDimsPos());
  SmallVector<int64_t> interchangeVector =
      computeInterchangeFromDimPos(dimsToBlock, getOutputRank());
  SmallVector<Value> interchangedInputIvsPointLoops = inputIvsPointLoops;
  interchangedInputIvsPointLoops = interchange<Value>(
      interchangedInputIvsPointLoops, interchangeVector, /*offset=*/0);

  llvm::append_range(inputIvs, interchangedInputIvsPointLoops);
  Value scalar = builder.create<memref::LoadOp>(loc, getInput(), inputIvs);
  builder.create<memref::StoreOp>(loc, scalar, getOutput(), ivs);
  return success();
}

// TODO: (lorenzo) implement `reifyResultShapes`. Should we also derive
// `inferResultType` from this method?
LogicalResult
UnPackOp::reifyResultShapes(OpBuilder &builder,
                            ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(getOperation());
  Location loc = getLoc();

  DenseMap<int64_t, OpFoldResult> dimAndTileMapping = getDimAndTileMapping();
  auto buildOutputDim = [&](OpBuilder &builder, size_t dimIdx) -> OpFoldResult {
    ArrayRef<int64_t> outputShape = getOutputShape();
    if (!ShapedType::isDynamic(outputShape[dimIdx])) {
      return builder.getI64IntegerAttr(outputShape[dimIdx]);
    }
    // OpFoldResult dimBound = getDim(builder, loc, getInput(), dimIdx);
    // if (dimAndTileMapping.count(dimIdx)) {
    //   AffineExpr dim = builder.getAffineSymbolExpr(0);
    //   AffineExpr tile = builder.getAffineSymbolExpr(1);
    //   dimBound = makeComposedFoldedAffineApply(
    //       builder, loc, dim + tile,
    //       ArrayRef<OpFoldResult>{dimBound, dimAndTileMapping[dimIdx]});
    // }
    OpFoldResult dimBound = getDim(builder, loc, getOutput(), dimIdx);
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

SmallVector<Range> UnPackOp::getIterationDomain(OpBuilder &builder) {
  return ::getIterationDomain(*this, builder);
}

/// Infer result/output type given the input and the tile sizes.
ShapedType UnPackOp::inferResultType() {
  DenseMap<int64_t, OpFoldResult> tileAndPosMapping = getDimAndTileMapping();
  SmallVector<int64_t> inferredShape;
  inferredShape.reserve(getOutputRank());
  ShapedType inputType = getInputType();
  int64_t rank = getOutputRank();

  for (auto dim : llvm::seq<int64_t>(0, rank)) {
    if (tileAndPosMapping.count(dim)) {
      Optional<int64_t> maybeConstantTileSize =
          getConstantIntValue(tileAndPosMapping.lookup(dim));
      if (inputType.isDynamicDim(dim) || !maybeConstantTileSize) {
        inferredShape.push_back(ShapedType::kDynamicSize);
      } else {
        int64_t tile = *maybeConstantTileSize;
        // TODO: (lorenzo) We bail out if we don't have full tiles. But we can
        // also allow an output tensor with smaller dimensions. Example,
        // input: memref<2x8x8x2xf32> and output: memref<13x15xf32> with
        // dims_pos = [0, 1] and inner_tiles = [8, 2]. This is rejected today as
        // 8 and 2 do not fully divide 13 and 15. But this is actually a legit
        // operation. From the 16x16 we can extract 13x15 discarding previously
        // introduced padding values. We need to guard the extraction with an
        // if.
        int64_t sizeDim = inputType.getDimSize(dim) * tile;
        inferredShape.push_back(sizeDim);
      }
    } else {
      inferredShape.push_back(inputType.getShape()[dim]);
    }
  }
  assert(inferredShape.size() == getOutputRank() &&
         "expect inferredShape to match output rank");
  return TypeSwitch<Type, ShapedType>(inputType)
      .Case<RankedTensorType>([&](RankedTensorType t) -> ShapedType {
        return RankedTensorType::get(inferredShape, inputType.getElementType());
      })
      .Case<MemRefType>([&](MemRefType t) -> ShapedType {
        return MemRefType::get(inferredShape, inputType.getElementType());
      })
      .Default([&](Type t) {
        llvm_unreachable("unexpected type");
        return nullptr;
      });
}

LogicalResult UnPackOp::verify() {
  Operation *op = getOperation();
  size_t numberOfBlockingFactors = getMixedTiles().size();
  SmallVector<int64_t> dimsPos = extractFromI64ArrayAttr(getDimsPos());
  if (failed(commonVerifierPackAndUnPackOp(*this))) {
    return failure();
  }
  // Blocking factors must be less or equal than the output rank, and must
  // match the number of `dims_pos`.
  if (numberOfBlockingFactors > getOutputRank()) {
    return op->emitError(
        "blocking factors must be less or equal than the output rank");
  }
  // Require `dim_pos` to be in-bound. `dim_pos` carries the index of the
  // dimensions to block.
  if (!isInBound(dimsPos, getOutputRank()))
    return op->emitError("out-of-bound position");
  // Require input rank to match output rank + number of blocking factors.
  if ((getOutputRank() + numberOfBlockingFactors) != getInputRank()) {
    return op->emitError(
        "input rank must equal output rank + blocking factors");
  }
  // Bail out if the tile does not divide the dimension fully. In the case of
  // dynamic tile factors or dimensions, having a partial tile is undefined
  // behavior. TODO: (lorenzo): We could relax this if we allow to `undo` the
  // padding done in the pack operation. The product of dim(point_loop) and
  // dim(tile_loop) >= dim(input).
  if (areNotFullTiles(getOutputShape(), getDimAndTileMapping())) {
    return op->emitError(
        "invalid tile factor provided. Only full tiles are supported");
  }

  // Verify result type against inferred type.
  ShapedType expectedType = inferResultType();
  if (!isCompatible(expectedType, getOutputType())) {
    return op->emitError(
               "infered type do not match provided output type. Expected ")
           << expectedType << " but got: " << getOutputType();
  }
  return success();
}

SmallVector<utils::IteratorType> UnPackOp::getLoopIteratorTypes() {
  SmallVector<utils::IteratorType> iteratorTypes(getOutputRank(),
                                                 utils::IteratorType::parallel);
  return iteratorTypes;
}

#define GET_OP_CLASSES
#include "Standalone/Dialect/LinalgX/LinalgXOps.cpp.inc"
