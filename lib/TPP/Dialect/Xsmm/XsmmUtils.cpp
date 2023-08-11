//===- XsmmUtils.cpp ---------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Xsmm/XsmmUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir {
namespace xsmm {
namespace utils {

xsmm::DataTypeAttr getDataType(RewriterBase &rewriter, Type type) {
  auto elemType = getElementTypeOrSelf(type);
  if (elemType.isBF16())
    return xsmm::DataTypeAttr::get(rewriter.getContext(), xsmm::DataType::BF16);
  assert(elemType.isF32() && "Element type neither bf16 nor f32");
  return xsmm::DataTypeAttr::get(rewriter.getContext(), xsmm::DataType::F32);
}

} // namespace utils
} // namespace xsmm
} // namespace mlir
