// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TPP/Dialect/VNNI/VNNIDialect.h"

#include "TPP/Dialect/VNNI/VNNIOps.cpp.inc"
#include "TPP/Dialect/VNNI/VNNIOps.h"
#include "mlir/Parser/Parser.h"

namespace mlir {
namespace vnni {

VNNIDialect::VNNIDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<VNNIDialect>()) {
#define GET_OP_LIST
  addOperations<
#include "TPP/Dialect/VNNI/VNNIOps.cpp.inc"
      >();
}

} // namespace vnni
} // namespace mlir
