//===- MathOps.h - Math Ops ---------------------------------------*- C++-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MATHX_STANDALONE_OPS_H
#define MATHX_STANDALONE_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/VectorInterfaces.h"

//===----------------------------------------------------------------------===//
// Math Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Standalone/Dialect/Mathx/MathxOps.h.inc"

#endif // MATHX_STANDALONE_OPS_H
