//===- TppOps.cpp - Tpp dialect ops ----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Tpp/TppOps.h"
#include "TPP/Dialect/Tpp/TppDialect.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "TPP/Dialect/Tpp/TppOps.cpp.inc"

using namespace mlir;
using namespace mlir::tpp;

//===----------------------------------------------------------------------===//
// IdentityOp
//===----------------------------------------------------------------------===//

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
    return emitOpError("expects a shape type for output");
  unsigned rankOutput = outputType.cast<ShapedType>().getRank();
  if (rankOutput < rankInput)
    return emitOpError("expects output rank to be >= of input rank");

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
    return emitOpError("fails to verify broadcasting rules");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

static bool verifyMatmulShape(MemRefType memrefA, MemRefType memrefB,
                              MemRefType memrefC) {
  if (memrefB.getRank() != 2 || memrefC.getRank() != 2 ||
      memrefA.getRank() != 2)
    return false;
  return true;
}

static bool verifyMatmulOperandsDims(ArrayRef<int64_t> shapeA,
                                     ArrayRef<int64_t> shapeB,
                                     ArrayRef<int64_t> shapeC) {
  int64_t m = shapeC[0];
  int64_t n = shapeC[1];
  int64_t k = shapeA[1];
  // Verify C(m, n) = A(m, k) B(k, n)
  if (shapeB[0] != k || shapeB[1] != n || shapeA[0] != m)
    return false;
  return true;
}

// XXX: Changing the op semantics based on the type is so bad and brittle.
// We don't want to do this. This BF16 packing need to be revisited.
// Check that op to be 2d matmul in row-major.
LogicalResult MatmulOp::verify() {
  MemRefType memrefA = getMatrixA().getType().cast<MemRefType>();
  MemRefType memrefB = getMatrixB().getType().cast<MemRefType>();
  MemRefType memrefC = getMatrixC().getType().cast<MemRefType>();
  if (!verifyMatmulShape(memrefA, memrefB, memrefC))
    return emitOpError("fails to verify operands shapes");
  if (!verifyMatmulOperandsDims(memrefA.getShape(), memrefB.getShape(),
                                memrefC.getShape()))
    return emitOpError("fails to verify operands dimensions mismatch");
  return success();
}

void MatmulOp::build(OpBuilder &builder, OperationState &state,
                     ValueRange inputs, Value output) {
  MatmulOp::build(builder, state, inputs[0], inputs[1], output);
}

//===----------------------------------------------------------------------===//
// BrgemmOp
//===----------------------------------------------------------------------===//

static bool verifyBRGemmShape(MemRefType memrefA, MemRefType memrefB,
                              MemRefType memrefC) {
  if (memrefB.getRank() != 3 || memrefC.getRank() != 2 ||
      memrefA.getRank() != 3)
    return false;
  return true;
}

// XXX: Changing the op semantics based on the type is so bad and brittle.
// We don't want to do this. This BF16 packing need to be revisited.
LogicalResult BrgemmOp::verify() {
  MemRefType tensorA = getBatchMatrixA().getType().cast<MemRefType>();
  MemRefType tensorB = getBatchMatrixB().getType().cast<MemRefType>();
  MemRefType matrixC = getMatrixC().getType().cast<MemRefType>();
  if (!verifyBRGemmShape(tensorA, tensorB, matrixC))
    return emitOpError("fails to verify operands shapes");
  // Check batch dimension.
  if (tensorA.getShape()[0] != tensorB.getShape()[0])
    return emitOpError("fails to verify operands dimensions mismatch");
  // Check all others that must be 'matmul' like.
  if (!verifyMatmulOperandsDims(tensorA.getShape().drop_front(),
                                tensorB.getShape().drop_front(),
                                matrixC.getShape()))
    return emitOpError("fails to verify operands dimensions mismatch");
  return success();
}

void BrgemmOp::build(OpBuilder &builder, OperationState &state,
                     ValueRange inputs, Value output) {
  BrgemmOp::build(builder, state, inputs[0], inputs[1], output);
}

//===----------------------------------------------------------------------===//
// AdddOp
//===----------------------------------------------------------------------===//

// Accept only shaped operands for AddOp. We currently do not support
// broadcasting and TPP operations are memory to memory thus disallow scalar
// operand for now.
LogicalResult AddOp::verify() {
  Type lhsType = getLhs().getType();
  Type rhsType = getRhs().getType();
  if ((!lhsType.isa<ShapedType>()) || (!rhsType.isa<ShapedType>()))
    return emitOpError("expects both operands to be shaped type");
  return success();
}

//===----------------------------------------------------------------------===//
// VNNI_MatmulOp
//===----------------------------------------------------------------------===//

static bool verifyVNNIMatmulShape(MemRefType memrefA, MemRefType memrefB,
                                  MemRefType memrefC) {
  if (memrefB.getRank() != 2 || memrefC.getRank() != 2 ||
      memrefA.getRank() != 3)
    return false;
  return true;
}

static bool verifyVNNIMatmulOperandsDims(ArrayRef<int64_t> shapeA,
                                         ArrayRef<int64_t> shapeB,
                                         ArrayRef<int64_t> shapeC) {
  int64_t m = shapeC[0];
  int64_t n = shapeC[1];
  int64_t k = shapeA[1];
  if (shapeB[0] != k || shapeB[1] != n || shapeA[0] * shapeA[2] != m)
    return false;
  return true;
}

LogicalResult VNNI_MatmulOp::verify() {
  MemRefType memrefA = getMatrixA().getType().cast<MemRefType>();
  MemRefType memrefB = getMatrixB().getType().cast<MemRefType>();
  MemRefType memrefC = getMatrixC().getType().cast<MemRefType>();
  assert(memrefA.getElementType().isBF16() && memrefA.getRank() == 3);
  if (!verifyVNNIMatmulShape(memrefA, memrefB, memrefC))
    return emitOpError("fails to verify operands shapes");
  if (!verifyVNNIMatmulOperandsDims(memrefA.getShape(), memrefB.getShape(),
                                    memrefC.getShape()))
    return emitOpError("fails to verify operands dimensions mismatch");
  return success();
}

void VNNI_MatmulOp::build(OpBuilder &builder, OperationState &state,
                          ValueRange inputs, Value output) {
  VNNI_MatmulOp::build(builder, state, inputs[0], inputs[1], output);
}

//===----------------------------------------------------------------------===//
// BrgemmOp
//===----------------------------------------------------------------------===//

static bool verifyVNNIBRGemmShape(MemRefType memrefA, MemRefType memrefB,
                                  MemRefType memrefC) {
  if (memrefB.getRank() != 3 || memrefC.getRank() != 2 ||
      memrefA.getRank() != 4)
    return false;
  return true;
}

LogicalResult VNNI_BrgemmOp::verify() {
  MemRefType tensorA = getBatchMatrixA().getType().cast<MemRefType>();
  MemRefType tensorB = getBatchMatrixB().getType().cast<MemRefType>();
  MemRefType matrixC = getMatrixC().getType().cast<MemRefType>();
  if (!verifyVNNIBRGemmShape(tensorA, tensorB, matrixC))
    return emitOpError("fails to verify operands shapes");
  // Check batch dimension.
  if (tensorA.getShape()[0] * tensorA.getShape()[3] != tensorB.getShape()[0])
    return emitOpError("fails to verify operands dimensions mismatch");
  return success();
}

void VNNI_BrgemmOp::build(OpBuilder &builder, OperationState &state,
                          ValueRange inputs, Value output) {
  VNNI_BrgemmOp::build(builder, state, inputs[0], inputs[1], output);
}
