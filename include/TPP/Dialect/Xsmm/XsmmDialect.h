//===- XsmmDialect.h - Xsmm dialect -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_DIALECT_XSMM_XSMMDIALECT_H
#define TPP_DIALECT_XSMM_XSMMDIALECT_H

#include "mlir/IR/Dialect.h"

#define GET_OP_CLASSES
#include "TPP/Dialect/Xsmm/XsmmOpsDialect.h.inc"

#endif // TPP_DIALECT_XSMM_XSMMDIALECT_H
