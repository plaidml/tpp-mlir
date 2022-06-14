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
/*
using namespace mlir;
using namespace mlir::tpp;

// TODO: For some reason hasVerifier = 1 does not add
// the hook.
LogicalResult IdentityOp::verify() {
  1. The input can be a constant, 1d and 2 memref.
  2. The output must no be a constant, but only
  a 1d/2d memref.
}
*/
