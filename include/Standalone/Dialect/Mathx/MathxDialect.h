//===- MathxDialect.h - Mathx dialect ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MATHX_STANDALONE_DIALECT_H
#define MATHX_STANDALONE_DIALECT_H

// clang-format off
// XXX: Dialect should be included before *.inc
#include "mlir/IR/Dialect.h"

#include "Standalone/Dialect/Mathx/MathxOpsDialect.h.inc"
// clang-format on

#endif // MATHX_STANDALONE_DIALECT_H
