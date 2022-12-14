//===- PerfTypes.h - Perf Dialect Types -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the types for the Perf dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TPP_DIALECT_PERF_PERFTYPES_H
#define TPP_DIALECT_PERF_PERFTYPES_H

#include "mlir/IR/Types.h"

//===----------------------------------------------------------------------===//
// Perf Dialect Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "TPP/Dialect/Perf/PerfOpsTypes.h.inc"

#endif // TPP_DIALECT_PERF_PERFTYPES_H
