//===- TransformUtils.cpp ----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace mlir {

class OpBuilder;
struct Range;
class RewriterBase;

namespace linalg {
class LinalgOp;
}

namespace utils {

// Given an opOperand and a range of ivs return the one used by the operands.
FailureOr<SmallVector<Value>>
getInvolvedLocalDimsForOperand(OpBuilder &builder, Location loc,
                               OpOperand *operand, AffineMap mapOperand,
                               ValueRange localIvs);

// Extract and return a slice for operand. Offsets are the induction variable
// touched by the operand. Sizes are: '1' in [0 to rank - desiredResultRank]
// while the full chunk in [rank - desiredResultRank to rank). Strides are
// assumed to be always 1. The methods effectively peel out the outermost [0 to
// rank - desiredResultRank] dimensions that are materialized as loops.
FailureOr<Value> getSliceOperand(OpBuilder &builder, OpOperand *operand,
                                 linalg::LinalgOp linalgOp, ValueRange ivs,
                                 ValueRange valuesToUse,
                                 unsigned desiredResultRank);

// Extract a slice of `operand` based on `offset`, `sizes` and
// `strides`.
Value getSliceOperand(OpBuilder &builder, linalg::LinalgOp linalgOp,
                      Value operand, ArrayRef<OpFoldResult> offset,
                      ArrayRef<OpFoldResult> sizes,
                      ArrayRef<OpFoldResult> strides,
                      unsigned desiredResultRank);

// Return the loop range to materialize as loops from '0' to 'upTo'.
// '0' is the outermost loop.
FailureOr<SmallVector<Range>> getLoopsToMaterialize(RewriterBase &rewriter,
                                                    linalg::LinalgOp linalgOp,
                                                    unsigned upTo);

} // namespace utils

} // namespace mlir
