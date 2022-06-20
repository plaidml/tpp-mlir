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

  // check if the shape are compatible.
  ArrayRef<int64_t> shapeInput = inputType.cast<ShapedType>().getShape();
  ArrayRef<int64_t> shapeOutput = outputType.cast<ShapedType>().getShape();

  if (rankInput == 1 && rankOutput == 2 && shapeInput[0] == shapeOutput[1])
    return success();

  assert(rankInput == rankOutput && "expect same rank");
  for (int idx = rankInput - 1; idx >= 0; idx--)
    if (shapeInput[idx] != shapeOutput[idx])
      return emitError("incompatible shape");
  return success();
}

// Check that op to be 2d matmul in row-major.
LogicalResult MatmulOp::verify() {
  MemRefType A = getMatrixA().getType().cast<MemRefType>();
  MemRefType B = getMatrixB().getType().cast<MemRefType>();
  MemRefType C = getMatrixC().getType().cast<MemRefType>();
  if ((A.getShape().size() != 2) || (B.getShape().size() != 2) ||
      (C.getShape().size() != 2))
    return failure();
  int64_t m = C.getShape()[0];
  int64_t n = C.getShape()[1];
  int64_t k = A.getShape()[1];
  if ((A.getShape()[0] != m) || (B.getShape()[1] != n) ||
      (B.getShape()[0] != k))
    return failure();
  return success();
}
