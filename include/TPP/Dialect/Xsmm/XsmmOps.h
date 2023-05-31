//===- XsmmOps.h - Xsmm dialect ops -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_DIALECT_XSMM_XSMMOPS_H
#define TPP_DIALECT_XSMM_XSMMOPS_H

#include "TPP/Dialect/Xsmm/XsmmDialect.h"
#include "TPP/Dialect/Xsmm/XsmmEnum.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "TPP/Dialect/Xsmm/XsmmOps.h.inc"

#endif // TPP_DIALECT_XSMM_XSMMOPS_H
