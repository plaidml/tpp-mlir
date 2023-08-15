//===- ValueUtils.h - -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_VALUE_UTILS_H
#define TPP_VALUE_UTILS_H

namespace mlir {
class Value;
namespace utils {

// Returns true if the value is a constant float or integer.
bool isValConstZero(Value val);

// Returns true if the op defining `val` represents a zero filled tensor.
bool isZeroTensor(Value val);

} // namespace utils
} // namespace mlir

#endif // TPP_VALUE_UTILS_H
