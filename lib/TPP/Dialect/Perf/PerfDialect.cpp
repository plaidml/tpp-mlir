//===- PerfDialect.cpp - Perf dialect ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Perf/PerfDialect.h"
#include "TPP/Dialect/Perf/PerfOps.h"

using namespace mlir;
using namespace mlir::perf;

//===----------------------------------------------------------------------===//
// Perf dialect.
//===----------------------------------------------------------------------===//

void PerfDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TPP/Dialect/Perf/PerfOps.cpp.inc"
      >();
}

#include "TPP/Dialect/Perf/PerfOpsDialect.cpp.inc"
