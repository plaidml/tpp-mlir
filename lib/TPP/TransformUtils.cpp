//===- TransformUtils.cpp ----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;

namespace mlir {

namespace utils {

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
          makeComposedAffineApply(builder, loc, resMap, touchedIvs)
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
  SmallVector<OpFoldResult> domain = makeComposedFoldedMultiResultAffineApply(
      rewriter, loc, map, allShapeSizes);
  SmallVector<Range> loopRanges;
  for (unsigned idx = 0; idx < upTo; idx++)
    loopRanges.push_back(
        Range{rewriter.getIndexAttr(0), domain[idx], rewriter.getIndexAttr(1)});
  return loopRanges;
}

} // namespace utils

} // namespace mlir
