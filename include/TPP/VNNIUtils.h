//===- VNNIUtils.cpp ---------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_VNNIUTILS_H
#define TPP_VNNIUTILS_H

#include <cstdint>
#include <optional>

namespace mlir {
class Type;
class MemRefType;

namespace vnni {
namespace utils {

// Returns the VNNI blocking factor: 2 for BF16 and 4 for BF8.
std::optional<int64_t> getVnniBlockingFactor(Type type);

// Returns true if the type is BF16.
bool isBF16Type(Type type);

// Return true if the memref is in VNNI layout.
bool isInVnniLayout(MemRefType memref);

} // namespace utils
} // namespace vnni
} // namespace mlir

#endif
