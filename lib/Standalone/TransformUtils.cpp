//===- TransformUtils.cpp ----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
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

static SmallVector<int64_t>
getExpectedResultMemRefShape(SmallVector<OpFoldResult> sizes,
                             unsigned desiredResultRank) {
  SmallVector<int64_t> targetShape;
  int toSkip = sizes.size() - desiredResultRank;
  assert(toSkip >= 0);
  // TODO: find better way to express `skipping the first `toSkip`
  // elements`. Also would be nice to have `inferRankReducedResultType`
  // for subview to have the same API has the one for tensor. This
  // would allow us to pass only `desiredResultRank` and avoid
  // this method.
  for (unsigned idx = 0, e = sizes.size(); idx < e; idx++) {
    if (toSkip > 0) {
      toSkip--;
      continue;
    }
    Optional<int64_t> sizeDim = getConstantIntValue(sizes[idx]);
    assert(sizeDim && "must be statically known");
    targetShape.push_back(sizeDim.value());
  }
  return targetShape;
}

Value getSlicedOperand(OpBuilder &builder, linalg::LinalgOp linalgOp,
                       Value operand, SmallVector<OpFoldResult> offsets,
                       SmallVector<OpFoldResult> sizes,
                       SmallVector<OpFoldResult> strides,
                       unsigned desiredResultRank) {
  ShapedType operandType = operand.getType().cast<ShapedType>();
  assert(operandType.hasStaticShape() && "tensor must have static shape");
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

  assert(extractOperation->getNumResults() == 1);
  return extractOperation->getResult(0);
}

} // namespace utils

} // namespace mlir
