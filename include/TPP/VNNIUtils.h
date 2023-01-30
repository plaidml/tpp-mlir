#ifndef TPP_VNNIUTILS_H
#define TPP_VNNIUTILS_H

//===- VNNIUtils.cpp ---------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace vnni {
namespace utils {
// Returns the VNNI blocking factor: 2 for BF16/4 for BF8.
Optional<int64_t> getVNNIBlockingFactor(Type type);

// Checks if the type is BF16.
bool isBF16Type(Type type);

} // namespace utils
} // namespace vnni
} // namespace mlir

#endif
