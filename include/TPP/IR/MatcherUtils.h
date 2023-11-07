//===- MatcherUtils.h - -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_MATCHER_UTILS_H
#define TPP_MATCHER_UTILS_H

namespace mlir {
class Value;
namespace linalg {
class LinalgOp;
} // namespace linalg
namespace structured_match {
namespace utils {

// Returns true if the linalg operation is a 2d eltwsie floating point addition.
bool isTwoDAddOp(linalg::LinalgOp linalgOp,
                 SmallVectorImpl<Value> *capturedOperands = nullptr);

// Returns true if the linalg operation is a 2d eltwsie floating point
// subtraction.
bool isTwoDSubOp(linalg::LinalgOp linalgOp,
                 SmallVectorImpl<Value> *capturedOperands = nullptr);

// Returns true if the linalg.generic is a 2d eltwise floating point fill
// operation with zeros.
bool isTwoDZeroOp(linalg::LinalgOp linalgOp,
                  SmallVectorImpl<Value> *capturedOperands = nullptr);

// Returns true if the linalg.generic is a 2d eltwise floating point relu
// operation.
bool isTwoDReluOp(linalg::LinalgOp linalgOp,
                  SmallVectorImpl<Value> *capturedOperands = nullptr);

// Returns true if the linalg.generic is a 2d floating point copy operation.
bool isTwoDIdentityOp(linalg::LinalgOp linalgOp,
                      SmallVectorImpl<Value> *capturedOperands = nullptr);

// Returns true if the linalg.generic can convert to a 2d eltwise floating point
// addition followed by a floating point relu.
bool isTwoDBiasReluOp(linalg::LinalgOp linalgOp,
                      SmallVectorImpl<Value> *capturedOperands = nullptr);

// Returns true if linalgOp is a 2d `linalg.transposeOp`.
bool isTwoDTransposeOp(linalg::LinalgOp linalgOp,
                       SmallVectorImpl<Value> *capturedOperands = nullptr);

// Return true if linalgOp is a 2d `linalgFillOp`. The fill is filling the
// output with zeros.
bool isTwoDFillOpWithZeros(linalg::LinalgOp linalgOp,
                           SmallVectorImpl<Value> *capturedOperands = nullptr);

} // namespace utils
} // namespace structured_match
} // namespace mlir

#endif // TPP_MATCHER_UTILS_H
