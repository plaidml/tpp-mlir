//===- StdxDialect.cpp - Stdx dialect ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Stdx/StdxDialect.h"
#include "TPP/Dialect/Stdx/StdxOps.h"

using namespace mlir;
using namespace mlir::stdx;

//===----------------------------------------------------------------------===//
// Tpp dialect.
//===----------------------------------------------------------------------===//

void StdxDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TPP/Dialect/Stdx/StdxOps.cpp.inc"
      >();
}

#include "TPP/Dialect/Stdx/StdxOpsDialect.cpp.inc"
