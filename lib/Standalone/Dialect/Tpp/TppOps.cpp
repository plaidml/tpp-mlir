//===- TppOps.cpp - Tpp dialect ops ----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/Tpp/TppOps.h"
#include "Standalone/Dialect/Tpp/TppDialect.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "Standalone/Dialect/Tpp/TppOps.cpp.inc"

using namespace mlir;
using namespace mlir::tpp;

LogicalResult IdentityOp::verify() {
  Type inputType = input().getType();
  Type outputType = output().getType();

  // input scalar, just return.
  if (!inputType.isa<ShapedType>())
    return success();

  // if the input is not a scalar the output rank should be >= of the input
  // rank.
  unsigned rankInput = inputType.cast<ShapedType>().getRank();
  if (!outputType.isa<ShapedType>())
    return failure();
  unsigned rankOutput = outputType.cast<ShapedType>().getRank();
  if (rankOutput >= rankInput)
    return success();
  return failure();
}
