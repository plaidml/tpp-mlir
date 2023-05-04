//===- VNNIOps.h - Check dialect ops ----------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef VNNI_TPP_OPS_H
#define VNNI_TPP_OPS_H

#include "TPP/Dialect/VNNI/VNNIDialect.h"
#include "TPP/Dialect/VNNI/VNNIInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "TPP/Dialect/VNNI/VNNIOps.h.inc"

#endif // VNNI_TPP_OPS_H
