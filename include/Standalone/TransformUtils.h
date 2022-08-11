//===- TransformUtils.cpp ----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace mlir {

class OpBuilder;

namespace utils {

// Given an opOperand and a range of ivs return the one used by the operands.
FailureOr<SmallVector<Value>>
getInvolvedLocalDimsForOperand(OpBuilder &builder, Location loc,
                               OpOperand *operand, AffineMap mapOperand,
                               ValueRange localIvs);

// Return a sliced operand using the localIvs as offset.
Value getSlicedOperand(OpBuilder &builder, Location loc, ValueRange localIvs,
                       linalg::LinalgOp linalgOp, OpOperand *operand,
                       ValueRange valuesToUse, unsigned desiredResultRank);

} // namespace utils

} // namespace mlir
