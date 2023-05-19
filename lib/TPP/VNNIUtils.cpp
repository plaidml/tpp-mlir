//===- VNNIUtils.cpp ---------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/VNNIUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace vnni {
namespace utils {

std::optional<int64_t> getVnniBlockingFactor(Type type) {
  auto elementType = getElementTypeOrSelf(type);
  if (!elementType.isBF16())
    return std::nullopt;
  // @ TODO we should call LIBXSMM here to query the 
  // VNNI packing factor we need to match to
#if defined(__x86_64__)
  return 2;
#elif defined(__aarch64__)
  return 4;
#else
#error Unsupported architecture
#endif
}

bool isBF16Type(Type type) {
  auto elementType = getElementTypeOrSelf(type);
  return elementType.isBF16();
}

bool isInVnniLayout(MemRefType memref) {
  if (memref.getRank() < 3 || !memref.getElementType().isBF16())
    return false;
  return memref.getShape()[memref.getRank() - 1] ==
         vnni::utils::getVnniBlockingFactor(memref);
}

} // namespace utils
} // namespace vnni
} // namespace mlir
