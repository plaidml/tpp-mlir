//===- TransformUtils.cpp ----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "TPP/IR/StructuredOpMatcher.h"
#include "TPP/TransformUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExprVisitor.h"

namespace mlir {

namespace linalgx {

namespace utils {

// taken from LinalgInterfaces.cpp
// Returns true if the use-def chain from `v` to `from` consists of 0 or more
// unary single-operand operations.
// TODO: relax to multi-operands with constants, which are technically unary ops
// as needed (e.g. add5).
static bool isChainOfUnaryOpsFrom(Value v, Value from) {
  while (true) {
    if (v == from)
      return true;
    Operation *op = v.getDefiningOp();
    if (!op || op->getNumOperands() != 1)
      return false;
    v = op->getOperand(0);
  };
}

// taken from LinalgInterfaces.cpp
// Returns the unique instance of OpType in `block` if it is indeed unique.
// Returns null if none or more than 1 instances exist.
template <typename OpType> static OpType getSingleOpOfType(Block &block) {
  OpType res = nullptr;
  block.walk([&](OpType op) {
    if (res) {
      res = nullptr;
      return WalkResult::interrupt();
    }
    res = op;
    return WalkResult::advance();
  });
  return res;
}

// Taken from: LinalgInterfaces.cpp
// Detect whether res is any permutation of `u5(u1(c) + u2(u3(a) * u4(b)))`
// on the field (AddOpType, MulOpType), where u1, u2, u3, u4 and u5 represent
// unary operations that may change the type.
template <typename AddOpType, typename MulOpType>
static bool isAddMul(linalg::LinalgOp linalgOp,
                     SmallVectorImpl<Value> *capturedOperands) {
  Block &block = linalgOp->getRegion(0).front();
  if (block.getNumArguments() != 3)
    return false;
  Operation *yieldOp = block.getTerminator();
  if (yieldOp->getNumOperands() != 1)
    return false;

  AddOpType addOp = getSingleOpOfType<AddOpType>(block);
  MulOpType mulOp = getSingleOpOfType<MulOpType>(block);
  if (!addOp || !mulOp)
    return false;

  BlockArgument argA = block.getArgument(0), argB = block.getArgument(1);
  Value a = mulOp->getOperand(0), b = mulOp->getOperand(1);
  Value mul = mulOp->getResult(0);
  BlockArgument argC = block.getArgument(2);
  Value c1 = addOp->getOperand(0), c2 = addOp->getOperand(1);
  Value add = addOp->getResult(0);
  Value res = yieldOp->getOperand(0);
  // Result traces back to add.
  auto un = isChainOfUnaryOpsFrom;
  bool success = un(res, add);
  // One of the operands of add traces back to argC, the other to the mul.
  success |= (un(c1, argC) && un(c2, mul)) || ((un(c1, mul)) && un(c2, argC));
  // One of the operands of mul traces back to argA, the other to argB.
  success |= (un(a, argA) && un(b, argB)) || ((un(a, argB)) && un(b, argA));
  if (capturedOperands) {
    capturedOperands->push_back(linalgOp.getMatchingOpOperand(argA)->get());
    capturedOperands->push_back(linalgOp.getMatchingOpOperand(argB)->get());
    capturedOperands->push_back(linalgOp.getMatchingOpOperand(argC)->get());
  }
  return success;
}

bool hasMulAddBody(linalg::LinalgOp linalgOp,
                   SmallVectorImpl<Value> *capturedOperands) {
  if (linalgOp->getNumRegions() != 1)
    return false;
  Region &region = linalgOp->getRegion(0);
  if (!region.hasOneBlock())
    return false;
  if (std::distance(region.front().begin(), region.front().end()) != 3)
    return false;
  bool isFloat =
      isAddMul<arith::AddFOp, arith::MulFOp>(linalgOp, capturedOperands);
  bool isInt =
      isAddMul<arith::AddIOp, arith::MulIOp>(linalgOp, capturedOperands);
  return (isFloat || isInt);
}

// Given localIvs being outermost dimensions of the current linalg operation,
// return the dimensions used by a given operand looking at its access map. As
// a simple example consider the following: map operand = (d0, d1, d2, d3, d4,
// d5, d6) -> (d0, d1 + d2, d4 + d3, d6) Assuming localIvs = (d0, d1, d2, d3)
// The result is: {d0, affine_apply(d1 + d2), d3}.
FailureOr<SmallVector<Value>>
getInvolvedLocalDimsForOperand(OpBuilder &builder, Location loc,
                               OpOperand *operand, AffineMap mapOperand,
                               ValueRange localIvs) {
  if (mapOperand.getNumSymbols() != 0)
    return failure();
  SmallVector<Value> ivsResult;
  ArrayRef<AffineExpr> results = mapOperand.getResults();
  for (size_t idx = 0, e = results.size(); idx < e; idx++) {
    AffineMap resMap = compressUnusedDims(mapOperand.getSubMap(idx));
    SmallVector<Value> touchedIvs;
    for (unsigned pos = 0, e = localIvs.size(); pos < e; pos++) {
      if (results[idx].isFunctionOfDim(pos))
        touchedIvs.push_back(localIvs[pos]);
    }
    // operand does not use any of the 'localIvs', keep going.
    if (touchedIvs.size() == 0)
      continue;
    if (touchedIvs.size() > 1) {
      // touched ivs should equal the number of dimensions.
      // if this is not the case, fail.
      if (resMap.getNumDims() != touchedIvs.size()) {
        resMap.dump();
        return failure();
      }
      ivsResult.push_back(
          affine::makeComposedAffineApply(builder, loc, resMap,
                                          getAsOpFoldResult(touchedIvs))
              .getResult());
    } else
      // single dimension touched just return it.
      ivsResult.push_back(touchedIvs[0]);
  }
  return ivsResult;
}

// Return the 'desiredResultRank' innermost subtensor dimensions.
// Example: sizes = {32, 64, 1, 23, 4} and desiredResultRank = 2.
// Result is {23, 4}.
// The method assumes the dimension to be statically known.
static SmallVector<int64_t>
getExpectedResultMemRefShape(ArrayRef<OpFoldResult> sizes,
                             unsigned desiredResultRank) {

  SmallVector<int64_t> targetShape;
  SmallVector<int64_t> sourceShapeStatic;
  SmallVector<Value> sourceShapeDynamic;
  dispatchIndexOpFoldResults(sizes, sourceShapeDynamic, sourceShapeStatic);

  // TODO: Would be nice to have `inferRankReducedResultType` for subview to
  // have the same API has the one for tensor. This would allow us to pass only
  // `desiredResultRank` and avoid this method.
  unsigned rank = sourceShapeStatic.size();
  unsigned currentSize = rank - desiredResultRank;
  for (unsigned idx = currentSize; idx < rank; idx++)
    targetShape.push_back(sourceShapeStatic[idx]);
  return targetShape;
}

// TODO: Check if we can merge with the function below `FailureOr<Value>
// getSliceOperand`.
Value getSliceOperand(OpBuilder &builder, linalg::LinalgOp linalgOp,
                      Value operand, ArrayRef<OpFoldResult> offsets,
                      ArrayRef<OpFoldResult> sizes,
                      ArrayRef<OpFoldResult> strides,
                      unsigned desiredResultRank) {
  ShapedType operandType = operand.getType().cast<ShapedType>();
  size_t rank = operandType.getRank();

  assert(rank == offsets.size() && "expect rank == offsets");
  assert(rank == sizes.size() && "expect rank == sizes");
  assert(rank == strides.size() && "expect rank == strides");

  Location loc = linalgOp.getLoc();
  Type reducedType =
      (linalgOp.hasTensorSemantics())
          ? tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
                desiredResultRank, operandType.cast<RankedTensorType>(),
                offsets, sizes, strides)
          : memref::SubViewOp::inferRankReducedResultType(
                getExpectedResultMemRefShape(sizes, desiredResultRank),
                operandType.cast<MemRefType>(), offsets, sizes, strides);

  Operation *extractOperation =
      (linalgOp.hasTensorSemantics())
          ? builder.create<tensor::ExtractSliceOp>(
                loc, reducedType.cast<RankedTensorType>(), operand, offsets,
                sizes, strides)
          : builder.create<memref::SubViewOp>(loc,
                                              reducedType.cast<MemRefType>(),
                                              operand, offsets, sizes, strides);

  assert(extractOperation->getNumResults() == 1 && "expect single result");
  return extractOperation->getResult(0);
}

static Value getSliceOperandImpl(OpBuilder &builder, linalg::LinalgOp linalgOp,
                                 OpOperand *operand, ValueRange ivs,
                                 ValueRange valuesToUse,
                                 unsigned desiredResultRank) {
  Value operandToUse = valuesToUse[operand->getOperandNumber()];
  ShapedType operandType = operandToUse.getType().cast<ShapedType>();
  size_t rank = operandType.getRank();
  // Happy path, use the current operand.
  if (rank == desiredResultRank)
    return operandToUse;

  SmallVector<OpFoldResult> offsets, sizes;
  offsets.reserve(rank);
  sizes.reserve(rank);

  // offset into the tensor is the induction var or 0.
  for (size_t idx = 0, e = ivs.size(); idx < e; idx++)
    offsets.push_back(ivs[idx]);
  for (size_t idx = ivs.size(), e = rank; idx < e; idx++)
    offsets.push_back(builder.getIndexAttr(0));

  // sizes are 1 in [0 to rank - desiredResultRank)
  // and 'full' in [rank - desiredResultRank to rank).
  for (size_t idx = 0, e = rank - desiredResultRank; idx < e; idx++)
    sizes.push_back(builder.getIndexAttr(1));
  for (size_t idx = rank - desiredResultRank, e = rank; idx < e; idx++)
    sizes.push_back(linalg::createFoldedDimOp(builder, linalgOp.getLoc(),
                                              operandToUse, idx));

  // strides are assumed to be always 1.
  SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));
  return utils::getSliceOperand(builder, linalgOp, operandToUse, offsets, sizes,
                                strides, desiredResultRank);
}

FailureOr<Value> getSliceOperand(OpBuilder &builder, OpOperand *operand,
                                 linalg::LinalgOp linalgOp, ValueRange ivs,
                                 ValueRange valuesToUse,
                                 unsigned desiredResultRank) {
  Location loc = linalgOp.getLoc();
  FailureOr<SmallVector<Value>> involvedDimForOperand =
      utils::getInvolvedLocalDimsForOperand(
          builder, loc, operand, linalgOp.getMatchingIndexingMap(operand), ivs);
  if (failed(involvedDimForOperand))
    return failure();
  return getSliceOperandImpl(builder, linalgOp, operand, *involvedDimForOperand,
                             valuesToUse, desiredResultRank);
}

FailureOr<SmallVector<Range>> getLoopsToMaterialize(RewriterBase &rewriter,
                                                    linalg::LinalgOp linalgOp,
                                                    unsigned upTo) {
  Location loc = linalgOp.getLoc();
  SmallVector<OpFoldResult> allShapeSizes =
      linalgOp.createFlatListOfOperandDims(rewriter, loc);
  AffineMap map = linalgOp.getShapesToLoopsMap();
  if (!map)
    return failure();
  SmallVector<OpFoldResult> domain =
      affine::makeComposedFoldedMultiResultAffineApply(rewriter, loc, map,
                                                       allShapeSizes);
  SmallVector<Range> loopRanges;
  for (unsigned idx = 0; idx < upTo; idx++)
    loopRanges.push_back(
        Range{rewriter.getIndexAttr(0), domain[idx], rewriter.getIndexAttr(1)});
  return loopRanges;
}

bool isBlockedConvolution(Operation *op) {
  if (!isa<linalg::LinalgOp>(op))
    return false;
  linalg::LinalgOp linalgOp = cast<linalg::LinalgOp>(op);
  auto iteratorTypes = linalgOp.getIteratorTypesArray();
  if (iteratorTypes.size() != 9)
    return false;
  bool match = linalg::isParallelIterator(iteratorTypes[0]) &&
               linalg::isParallelIterator(iteratorTypes[1]) &&
               linalg::isParallelIterator(iteratorTypes[2]) &&
               linalg::isParallelIterator(iteratorTypes[3]) &&
               linalg::isParallelIterator(iteratorTypes[4]) &&
               linalg::isReductionIterator(iteratorTypes[5]) &&
               linalg::isReductionIterator(iteratorTypes[6]) &&
               linalg::isReductionIterator(iteratorTypes[7]) &&
               linalg::isReductionIterator(iteratorTypes[8]);
  if (!match)
    return false;
  if (failed(mlir::linalg::detail::verifyConvolutionInterface(linalgOp)))
    return false;
  return hasMulAddBody(linalgOp, /*captures=*/nullptr);
}

FailureOr<linalg::ContractionDimensions>
isContraction(linalg::LinalgOp linalgOp) {
  using namespace structured_match;

  // clang-format off
  auto maybeContraction =
    StructuredOpMatcher::make<linalg::LinalgOp>()
    .operation(NumDpsInits(EqualsTo(1)))
    .operation(NumDpsInputs(EqualsTo(2)))
    .operation(NumAffineMaps(EqualsTo(3)))
    .region(MatchOne(0),
            WithOpChain<arith::MulFOp,
                        arith::AddFOp>(/*captures=*/nullptr));
  // clang-format on
  if (!maybeContraction.match(linalgOp))
    return failure();

  auto dims = linalg::inferContractionDims(linalgOp);
  if (failed(dims) ||
      (dims->m.size() < 1 || dims->n.size() < 1 || dims->k.size() < 1)) {
    return failure();
  }
  return dims;
}

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

static bool validateFullTilesOnDim(TilingInterface tileOp,
                                   const OpFoldResult &tile, size_t dim) {
  OpBuilder builder(tileOp);
  OpBuilder::InsertionGuard guard(builder);
  SmallVector<Range> iterationDomain =
      cast<TilingInterface>(tileOp.getOperation()).getIterationDomain(builder);
  if (dim >= iterationDomain.size())
    return false;

  auto tileFactor = getConstantIntValue(tile);
  auto rangeOnDim = getConstantRange(iterationDomain[dim]);

  // If the tile factor or the range are non-constant, the tile size is
  // considered to be valid.
  if (!tileFactor || !rangeOnDim)
    return true;

  // Corner case: Tiling with '0' along 'dim' is valid - no tiling.
  if (*tileFactor == 0)
    return true;

  // Corner case: Tiling '1' with '1' is valid.
  if (*tileFactor == 1 && *rangeOnDim == 1)
    return true;

  return (*rangeOnDim % *tileFactor == 0 && *rangeOnDim / *tileFactor != 1);
}

bool validateFullTilesOnDims(TilingInterface tileOp,
                             ArrayRef<OpFoldResult> tiles,
                             ArrayRef<size_t> dims) {
  if (!dims.empty() && dims.size() != tiles.size())
    return false;

  // If dims is empty we start from the outermost dim '0'.
  SmallVector<size_t> dimsToCheck;
  if (dims.empty())
    dimsToCheck = llvm::to_vector(llvm::seq<size_t>(0, tiles.size()));
  else
    dimsToCheck = llvm::to_vector(dims);
  assert(dimsToCheck.size() == tiles.size());

  for (auto dim : llvm::enumerate(dimsToCheck)) {
    if (!validateFullTilesOnDim(tileOp, tiles[dim.index()], dim.value()))
      return false;
  }
  return true;
}

namespace {

// Convert scf.for to scf.forall after fusion.
struct ConvertToForAll : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    auto metadata =
        forOp->getAttrOfType<StringAttr>(linalgx::utils::kLoopParallel);
    if (!metadata || metadata.getValue() != linalgx::utils::kLoopRoot)
      return failure();
    if (forOp.getNumRegionIterArgs() != 1)
      return failure();

    SmallVector<scf::ForOp> nestedLoops;
    getPerfectlyNestedLoops(nestedLoops, forOp);
    if (nestedLoops.size() == 0)
      return failure();

    SmallVector<Value> loopArgs;
    SmallVector<OpFoldResult> lbs, ubs, steps;
    scf::ForOp innerMostLoop = nestedLoops[nestedLoops.size() - 1];
    for (scf::ForOp &currentLoop : nestedLoops) {
      if (currentLoop.getNumRegionIterArgs() != 1)
        return failure();
      loopArgs.push_back(currentLoop.getInductionVar());
      lbs.push_back(currentLoop.getLowerBound());
      ubs.push_back(currentLoop.getUpperBound());
      steps.push_back(currentLoop.getStep());
      if (currentLoop == innerMostLoop) {
        // We can only replace if the last operation before the terminator is
        // an insert slice.
        auto yieldOp =
            cast<scf::YieldOp>(currentLoop.getBody()->getTerminator());
        auto insertSlice =
            yieldOp.getOperands()[0].getDefiningOp<tensor::InsertSliceOp>();
        if (!insertSlice)
          return failure();
        loopArgs.push_back(currentLoop.getRegionIterArg(0));
      }
    }

    rewriter.replaceOpWithNewOp<scf::ForallOp>(
        forOp, lbs, ubs, steps, ValueRange{forOp.getInitArgs()},
        /*mapping=*/std::nullopt,
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange regionArgs) {
          IRMapping mapping;
          assert(loopArgs.size() == regionArgs.size() &&
                 "expect same region args");
          mapping.map(loopArgs, regionArgs);
          Block *innerLoopBlock = nestedLoops[nestedLoops.size() - 1].getBody();
          auto yieldOp = cast<scf::YieldOp>(innerLoopBlock->getTerminator());
          auto insertSlice =
              yieldOp.getOperands()[0].getDefiningOp<tensor::InsertSliceOp>();
          assert(insertSlice && "must be an insert slice");
          for (auto &nestedOp : innerLoopBlock->without_terminator()) {
            if (&nestedOp == insertSlice.getOperation()) {
              auto term = nestedBuilder.create<scf::InParallelOp>(loc);
              nestedBuilder.setInsertionPointToStart(term.getBody());
              Value sourceVal = mapping.lookup(insertSlice.getSource());
              Value destVal = mapping.lookup(insertSlice.getDest());
              SmallVector<OpFoldResult> offsets;
              for (OpFoldResult offset : insertSlice.getMixedOffsets()) {
                if (auto valueOffset = offset.dyn_cast<Value>())
                  offsets.push_back(mapping.lookupOrDefault(valueOffset));
                else
                  offsets.push_back(offset);
              }
              SmallVector<OpFoldResult> sizes;
              for (OpFoldResult size : insertSlice.getMixedSizes()) {
                if (auto valueSize = size.dyn_cast<Value>())
                  sizes.push_back(mapping.lookupOrDefault(valueSize));
                else
                  sizes.push_back(size);
              }
              SmallVector<OpFoldResult> strides;
              for (OpFoldResult stride : insertSlice.getMixedStrides()) {
                if (auto valueStride = stride.dyn_cast<Value>())
                  strides.push_back(mapping.lookupOrDefault(valueStride));
                else
                  strides.push_back(stride);
              }
              assert(offsets.size() == sizes.size());
              assert(offsets.size() == strides.size());

              nestedBuilder.create<tensor::ParallelInsertSliceOp>(
                  loc, sourceVal, destVal, offsets, sizes, strides);
              continue;
            }
            Operation *clone = nestedBuilder.clone(nestedOp, mapping);
            mapping.map(nestedOp.getResults(), clone->getResults());
          }
        });
    return success();
  }
};

} // namespace

// Populate patterns to rewrite scf.for with scf.forall.
void populateScfForToForAllRewritePattern(RewritePatternSet &patterns) {
  patterns.add<ConvertToForAll>(patterns.getContext());
}

} // namespace utils

} // namespace linalgx

} // namespace mlir
