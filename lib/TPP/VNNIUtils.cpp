//===- VNNIUtils.cpp ---------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/VNNIUtils.h"

namespace mlir {
namespace vnni {
namespace utils {

Optional<int64_t> getVNNIBlockingFactor(Type type) {
  if (!isa<ShapedType>(type) ||
      !type.cast<ShapedType>().getElementType().isBF16())
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
  if (!isa<ShapedType>(type))
    return false;
  return type.cast<ShapedType>().getElementType().isBF16();
}
} // namespace utils
} // namespace vnni
} // namespace mlir
