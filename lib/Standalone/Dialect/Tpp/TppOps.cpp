//===- TppOps.cpp - Tpp dialect ops ----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/Tpp/TppOps.h"
#include "Standalone/Dialect/Tpp/TppDialect.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "Standalone/Dialect/Tpp/TppOps.cpp.inc"

using namespace mlir;
using namespace mlir::tpp;

LogicalResult IdentityOp::verify() {
  Type inputType = getInput().getType();
  Type outputType = getOutput().getType();

  // input scalar, just return.
  if (!inputType.isa<ShapedType>())
    return success();

  // if the input is not a scalar the output rank should be >= of the input
  // rank.
  unsigned rankInput = inputType.cast<ShapedType>().getRank();
  if (!outputType.isa<ShapedType>())
    return emitError("expect a shape type for output");
  unsigned rankOutput = outputType.cast<ShapedType>().getRank();
  if (rankOutput < rankInput)
    return emitError("output rank must be >= of input rank");

  // check if the shape are broadcast compatible.
  ArrayRef<int64_t> shapeInput = inputType.cast<ShapedType>().getShape();
  ArrayRef<int64_t> shapeOutput = outputType.cast<ShapedType>().getShape();

  for (int64_t i = rankInput - 1, j = rankOutput - 1; i >= 0 && j >= 0;
       i--, j--) {
    int64_t inputDim = shapeInput[i];
    int64_t outputDim = shapeOutput[j];

    if (inputDim == outputDim)
      continue;
    if (inputDim == 1 && outputDim > 1)
      continue;
    return emitError("broadcast incompatible");
  }
  return success();
}

// Check that op to be 2d matmul in row-major.
LogicalResult MatmulOp::verify() {
  MemRefType a = getMatrixA().getType().cast<MemRefType>();
  MemRefType b = getMatrixB().getType().cast<MemRefType>();
  MemRefType c = getMatrixC().getType().cast<MemRefType>();
  if ((a.getShape().size() != 2 && !a.getElementType().isBF16()) ||
      (a.getElementType().isBF16() && a.getShape().size() != 3 &&
       a.getShape().size() != 2) ||
      (b.getShape().size() != 2) || (c.getShape().size() != 2))
    return emitError("shapes incompatible");
  int64_t m = c.getShape()[0];
  int64_t n = c.getShape()[1];
  int64_t k = a.getShape()[1];
  if ((a.getShape().size() == 2 && a.getShape()[0] != m) ||
      (a.getShape().size() == 3 && a.getShape()[0] * a.getShape()[2] != m) ||
      (b.getShape()[1] != n) || (b.getShape()[0] != k))
    return emitError("Dimensions mismatching");
  return success();
}
