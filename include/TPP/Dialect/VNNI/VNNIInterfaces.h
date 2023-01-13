//===- VNNIInterface.h - VNNI operations interfaces -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the operation interfaces for VNNI operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_VNNI_IR_VNNIINTERFACES_H_
#define MLIR_DIALECT_VNNI_IR_VNNIINTERFACES_H_

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OpDefinition.h"

/// Include the generated interface declarations.
#include "TPP/Dialect/VNNI/VNNIInterfaces.h.inc"

#endif // MLIR_DIALECT_VNNI_IR_VNNIINTERFACES_H_
