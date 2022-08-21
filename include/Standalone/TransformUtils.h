//===- TransformUtils.cpp ----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/OpDefinition.h"

namespace mlir {

class OpBuilder;

namespace utils {

// Given an opOperand and a range of ivs return the one used by the operands.
FailureOr<SmallVector<Value>>
getInvolvedLocalDimsForOperand(OpBuilder &builder, Location loc,
                               OpOperand *operand, AffineMap mapOperand,
                               ValueRange localIvs);

// Extract and return a slice for operand using offsets, sizes, strides.
Value getSlicedOperand(OpBuilder &builder, linalg::LinalgOp linalgOp,
                       Value operand, SmallVector<OpFoldResult> offsets,
                       SmallVector<OpFoldResult> sizes,
                       SmallVector<OpFoldResult> strides,
                       unsigned desiredResultRank);

} // namespace utils

} // namespace mlir
