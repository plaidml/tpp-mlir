//===- XsmmDialect.cpp - Xsmm dialect ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Xsmm/XsmmDialect.h"
#include "TPP/Dialect/Xsmm/XsmmOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::xsmm;

//===----------------------------------------------------------------------===//
// Xsmm dialect.
//===----------------------------------------------------------------------===//

void XsmmDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TPP/Dialect/Xsmm/XsmmOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "TPP/Dialect/Xsmm/XsmmAttr.cpp.inc"
      >();
}

#include "TPP/Dialect/Xsmm/XsmmOpsDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "TPP/Dialect/Xsmm/XsmmAttr.cpp.inc"
