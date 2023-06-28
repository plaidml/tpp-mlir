//===- LinalgXTransformOps.h - Linalg transform ops -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALGX_TRANSFORMOPS_LINALGTRANSFORMOPS_H
#define MLIR_DIALECT_LINALGX_TRANSFORMOPS_LINALGTRANSFORMOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir {
namespace linalg {
class GenericOp;
class LinalgOp;
} // namespace linalg
} // namespace mlir

//===----------------------------------------------------------------------===//
// LinalgX Transform Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "TPP/Dialect/Transform/LinalgXTransformOps.h.inc"

namespace mlir {
namespace linalgx {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace linalgx
} // namespace mlir

#endif // MLIR_DIALECT_LINALGX_TRANSFORMOPS_LINALGTRANSFORMOPS_H
