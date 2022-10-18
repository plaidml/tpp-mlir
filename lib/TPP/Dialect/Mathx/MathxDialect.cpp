//===- MathxDialect.cpp - Mathx dialect -------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Mathx/MathxDialect.h"
#include "TPP/Dialect/Mathx/MathxOps.h"

using namespace mlir;
using namespace mlir::mathx;

//===----------------------------------------------------------------------===//
// Mathx dialect.
//===----------------------------------------------------------------------===//

void MathxDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TPP/Dialect/Mathx/MathxOps.cpp.inc"
      >();
}

#include "TPP/Dialect/Mathx/MathxOpsDialect.cpp.inc"
