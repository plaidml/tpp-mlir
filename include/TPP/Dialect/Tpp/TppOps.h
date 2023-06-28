//===- TppOps.h - Tpp dialect ops -------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_DIALECT_TPP_TPPOPS_H
#define TPP_DIALECT_TPP_TPPOPS_H

#include "TPP/Dialect/Tpp/TppAttr.h"
#include "TPP/Dialect/Tpp/TppDialect.h"
#include "TPP/Dialect/Tpp/TppInterface.h"
#include "TPP/Dialect/Tpp/TppTraits.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "TPP/Dialect/Tpp/TppOps.h.inc"

#endif // TPP_DIALECT_TPP_TPPOPS_H
