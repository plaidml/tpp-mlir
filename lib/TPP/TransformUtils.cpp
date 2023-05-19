//===- TransformUtils.cpp ----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "TPP/IR/StructuredOpMatcher.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
          affine::makeComposedAffineApply(builder, loc, resMap, touchedIvs)
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

static llvm::SmallDenseSet<unsigned> getPreservedDims(AffineMap map) {
  assert(map.isProjectedPermutation() &&
         "expected map to have projected permutations");
  llvm::SmallDenseSet<unsigned> preservedDims;
  for (auto expr : map.getResults())
    preservedDims.insert(expr.cast<AffineDimExpr>().getPosition());
  return preservedDims;
}

namespace {

// Walk the indexing expressions for the B operand in a blocked matmul and
// verify it is in the right form, either:
// - AffineDimExpr
// - AffineDimExpr floorDiv constant.
// Each dimension occurs only once.
struct OperandBExprWalker
    : public AffineExprVisitor<OperandBExprWalker, LogicalResult> {
  OperandBExprWalker() = delete;
  OperandBExprWalker(ArrayRef<mlir::utils::IteratorType> iteratorTypes)
      : iteratorTypes(iteratorTypes) {}
  llvm::SmallDenseSet<unsigned> bDims;

  LogicalResult visitDimExpr(AffineDimExpr dimExpr) {
    unsigned position = dimExpr.getPosition();
    if (bDims.count(position))
      return failure();
    bDims.insert(position);
    return success();
  }

  LogicalResult visitSymbolExpr(AffineSymbolExpr expr) { return failure(); }

  LogicalResult visitConstantExpr(AffineConstantExpr expr) { return failure(); }

  LogicalResult visitAffineBinaryOpExpr(AffineBinaryOpExpr binaryExpr) {
    if (binaryExpr.getKind() != AffineExprKind::FloorDiv)
      return failure();
    return success(succeeded(isDimExprOrCstExpr(binaryExpr.getLHS())) &&
                   succeeded(isDimExprOrCstExpr(binaryExpr.getRHS())));
  }

  LogicalResult isDimExprOrCstExpr(AffineExpr expr) {

    auto isReductionDim =
        [](unsigned position,
           ArrayRef<mlir::utils::IteratorType> iteratorTypes) -> bool {
      return iteratorTypes[position] == mlir::utils::IteratorType::reduction;
    };

    if (auto dimExpr = expr.dyn_cast<AffineDimExpr>()) {
      unsigned position = dimExpr.getPosition();
      if (bDims.count(position) || !isReductionDim(position, iteratorTypes))
        return failure();
      bDims.insert(position);
      return success();
    }
    if (auto cstExpr = expr.dyn_cast<AffineConstantExpr>())
      return success();
    return failure();
  }

private:
  ArrayRef<mlir::utils::IteratorType> iteratorTypes;
};

} // namespace

bool isBlockedMatmul(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp)
    return false;

  if (linalgOp.getNumDpsInputs() != 2 || linalgOp.getNumDpsInits() != 1)
    return false;

  if (!hasMulAddBody(linalgOp, /*captures=*/nullptr))
    return false;

  // Check the input indexing map has the right form.
  auto indexingMaps = linalgOp.getIndexingMapsArray();
  if (indexingMaps.size() != 3)
    return false;
  if ((!indexingMaps.back().isProjectedPermutation()) ||
      (!indexingMaps.front().isProjectedPermutation()))
    return false;
  // Walk the expressions for the B operand. The affine map may not be a
  // projected permutation if the blocked matmul is in VNNI format.
  auto iteratorTypes = linalgOp.getIteratorTypesArray();
  OperandBExprWalker operandBWalker(iteratorTypes);
  if (llvm::any_of(indexingMaps[1].getResults(),
                   [&operandBWalker](AffineExpr expr) {
                     return failed(operandBWalker.visit(expr));
                   })) {
    return false;
  }

  // Make sure all loops are characterized as one of:
  // - I loop: present in C and A but not in B. I must be parallel.
  // - J loop: present in C and B but not in A. J must be parallel.
  // - K loop: present in A and B but not in C. K must be reduction.
  llvm::SmallDenseSet<unsigned> cDims = getPreservedDims(indexingMaps.back());
  llvm::SmallDenseSet<unsigned> aDims = getPreservedDims(indexingMaps.front());
  llvm::SmallDenseSet<unsigned> allLoopDims;
  for (auto cExpr : indexingMaps.back().getResults()) {
    unsigned cDim = cExpr.cast<AffineDimExpr>().getPosition();
    // I loop
    if (aDims.count(cDim) && !operandBWalker.bDims.count(cDim)) {
      if (iteratorTypes[cDim] != mlir::utils::IteratorType::parallel)
        return false;
      allLoopDims.insert(cDim);
      continue;
    }
    // J loop
    if (operandBWalker.bDims.count(cDim) && !aDims.count(cDim)) {
      if (iteratorTypes[cDim] != mlir::utils::IteratorType::parallel)
        return false;
      allLoopDims.insert(cDim);
      continue;
    }
    return false;
  }

  for (auto aExpr : indexingMaps.front().getResults()) {
    unsigned aDim = aExpr.cast<AffineDimExpr>().getPosition();
    // I loop
    if (cDims.count(aDim) && !operandBWalker.bDims.count(aDim)) {
      // Already seen.
      continue;
    }
    // K loop
    if (operandBWalker.bDims.count(aDim) && !cDims.count(aDim)) {
      if (iteratorTypes[aDim] != mlir::utils::IteratorType::reduction)
        return false;
      allLoopDims.insert(aDim);
      continue;
    }
    return false;
  }

  // We may have an extra reduction in the B operand (i.e., VNNI).
  for (auto bDim : operandBWalker.bDims) {
    if (!allLoopDims.count(bDim)) {
      if (aDims.count(bDim) || cDims.count(bDim) ||
          iteratorTypes[bDim] != mlir::utils::IteratorType::reduction)
        return false;
      allLoopDims.insert(bDim);
    }
    continue;
  }

  // At this point we must have covered all the loops.
  return allLoopDims.size() == linalgOp.getNumLoops();
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

static bool isFullTile(int64_t tileFactor, int64_t range) {
  return range % tileFactor == 0;
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

  // Tiling with '0' along 'dim' is valid - no tiling.
  if (*tileFactor == 0)
    return true;

  return isFullTile(*tileFactor, *rangeOnDim);
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

  size_t idxInTiles = 0;
  for (size_t dim : dimsToCheck) {
    if (!validateFullTilesOnDim(tileOp, tiles[idxInTiles++], dim))
      return false;
  }
  return true;
}

namespace {
// Helper matcher functor for matmul detection.
struct WithMulAddBody {
  WithMulAddBody() = delete;
  WithMulAddBody(SmallVectorImpl<Value> *captures) : captures(captures){};

  bool operator()(Region *region, Operation *op) {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp)
      return false;

    return hasMulAddBody(linalgOp, captures);
  }

private:
  SmallVectorImpl<Value> *captures;
};
} // namespace

bool isMatmulOp(Operation *op, SmallVectorImpl<Value> *operands) {
  if (isa_and_nonnull<linalg::MatmulOp>(op))
    return true;
  if (!isa_and_nonnull<linalg::GenericOp>(op))
    return false;
  auto linalgOp = cast<linalg::GenericOp>(op);
  using namespace tpp::structured_match;
  using MapList = ArrayRef<ArrayRef<AffineExpr>>;
  auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
  AffineExpr i, j, k;
  bindDims(linalgOp.getContext(), i, j, k);
  auto mapList = infer({{i, k}, {k, j}, {i, j}});
  auto matmulMatcher =
      StructuredOpMatcher::make<linalg::GenericOp>()
          .operation(NumDpsInits(EqualsTo(1)))
          .operation(NumDpsInputs(EqualsTo(2)))
          .operation(NumRegions(EqualsTo(1)))
          .dim(MatchAll(), {mlir::utils::IteratorType::reduction,
                            mlir::utils::IteratorType::parallel,
                            mlir::utils::IteratorType::parallel})
          .input(MatchOne(0), HasMap(EqualsTo(mapList[0])))
          .input(MatchOne(1), HasMap(EqualsTo(mapList[1])))
          .output(MatchOne(0), HasMap(EqualsTo(mapList[2])))
          .region(MatchOne(0), WithMulAddBody(operands));
  return matmulMatcher.match(linalgOp);
}

} // namespace utils

} // namespace linalgx

} // namespace mlir
