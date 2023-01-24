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
int getVNNIBlockingFactor(Type type) {
  assert(type.cast<ShapedType>().getElementType().isBF16() &&
         "Only BF16 VNNI packing supported");
  return 2;
}
} // namespace utils
} // namespace vnni
} // namespace mlir
