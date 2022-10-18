//===- LinalgXDialect.cpp - LinalgX dialect ---------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/LinalgX/LinalgXDialect.h"
#include "TPP/Dialect/LinalgX/LinalgXOps.h"

using namespace mlir;
using namespace mlir::linalgx;

//===----------------------------------------------------------------------===//
// LinalgX dialect.
//===----------------------------------------------------------------------===//

void LinalgXDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TPP/Dialect/LinalgX/LinalgXOps.cpp.inc"
      >();
}

#include "TPP/Dialect/LinalgX/LinalgXOpsDialect.cpp.inc"
