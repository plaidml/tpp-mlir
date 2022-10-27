// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TPP/Dialect/Check/CheckDialect.h"

#include "TPP/Dialect/Check/CheckOps.cpp.inc"
#include "TPP/Dialect/Check/CheckOps.h"
#include "mlir/Parser/Parser.h"

namespace mlir {
namespace check {

CheckDialect::CheckDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<CheckDialect>()) {
#define GET_OP_LIST
  addOperations<
#include "TPP/Dialect/Check/CheckOps.cpp.inc"
      >();
}

} // namespace check
} // namespace mlir
