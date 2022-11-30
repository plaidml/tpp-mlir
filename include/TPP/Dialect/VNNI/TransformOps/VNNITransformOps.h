//===- VNNITransformOps.h - VNNI transform ops ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_VNNI_TRANSFORMOPS_VNNITRANSFORMOPS_H
#define MLIR_DIALECT_VNNI_TRANSFORMOPS_VNNITRANSFORMOPS_H

#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/OpImplementation.h"


//===----------------------------------------------------------------------===//
// VNNI Transform Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "TPP/Dialect/VNNI/TransformOps/VNNITransformOps.h.inc"

namespace mlir {
namespace vnni {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace vnni
} // namespace mlir

#endif // MLIR_DIALECT_VNNI_TRANSFORMOPS_VNNITRANSFORMOPS_H
