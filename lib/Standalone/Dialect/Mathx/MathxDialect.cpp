//===- MathxDialect.cpp - Mathx dialect -------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/Mathx/MathxDialect.h"
#include "Standalone/Dialect/Mathx/MathxOps.h"

using namespace mlir;
using namespace mlir::mathx;

//===----------------------------------------------------------------------===//
// Mathx dialect.
//===----------------------------------------------------------------------===//

void MathxDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Standalone/Dialect/Mathx/MathxOps.cpp.inc"
      >();
}
