//===- StdxDialect.cpp - Stdx dialect ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/Stdx/StdxDialect.h"
#include "Standalone/Dialect/Stdx/StdxOps.h"

using namespace mlir;
using namespace mlir::stdx;

//===----------------------------------------------------------------------===//
// Tpp dialect.
//===----------------------------------------------------------------------===//

void StdxDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Standalone/Dialect/Stdx/StdxOps.cpp.inc"
      >();
}
