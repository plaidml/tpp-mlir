//===- XsmmUtils.h - --------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef XSMM_DIALECT_XSMM_XSMMUTILS_H
#define XSMM_DIALECT_XSMM_XSMMUTILS_H

#include "TPP/Dialect/Xsmm/XsmmEnum.h"

namespace mlir {
class Type;
class RewriterBase;

namespace xsmm {
namespace utils {

xsmm::DataTypeAttr getDataType(RewriterBase &rewriter, Type type);

} // namespace utils
} // namespace xsmm
} // namespace mlir

#endif // XSMM_DIALECT_XSMM_XSMMUTILS_H
