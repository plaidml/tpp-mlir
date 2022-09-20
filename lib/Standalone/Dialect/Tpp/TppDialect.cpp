//===- TppDialect.cpp - Tpp dialect ----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/Tpp/TppDialect.h"
#include "Standalone/Dialect/Tpp/TppOps.h"

using namespace mlir;
using namespace mlir::tpp;

//===----------------------------------------------------------------------===//
// Tpp dialect.
//===----------------------------------------------------------------------===//

void TppDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Standalone/Dialect/Tpp/TppOps.cpp.inc"
      >();
}

#include "Standalone/Dialect/Tpp/TppOpsDialect.cpp.inc"
