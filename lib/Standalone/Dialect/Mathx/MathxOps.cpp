//===- MathxOps.cpp - Mathx dialect ops -------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/Mathx/MathxOps.h"
#include "Standalone/Dialect/Mathx/MathxDialect.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::mathx;

#define GET_OP_CLASSES
#include "Standalone/Dialect/Mathx/MathxOps.cpp.inc"

/// Materialize an integer or floating point constant.
Operation *mathx::MathxDialect::materializeConstant(OpBuilder &builder,
                                                    Attribute value, Type type,
                                                    Location loc) {
  return builder.create<arith::ConstantOp>(loc, value, type);
}
