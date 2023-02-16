//===- TppOps.cpp - Tpp dialect ops ----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Tpp/TppOps.h"
#include "TPP/Dialect/Tpp/TppDialect.h"
#include "TPP/Dialect/Tpp/TppUtils.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "TPP/Dialect/Tpp/TppOps.cpp.inc"

using namespace mlir;
using namespace mlir::tpp;

//===----------------------------------------------------------------------===//
// IdentityOp
//===----------------------------------------------------------------------===//

StringRef getMatchBroadcastRuleMessage(utils::MatchBroadcastRuleResult res) {
  switch (res) {
  case utils::MatchBroadcastRuleResult::OutputNotShapedType:
    return "expect shaped shaped type for output";
  case utils::MatchBroadcastRuleResult::WrongOutputRank:
    return "wrong rank on output";
  case utils::MatchBroadcastRuleResult::FailedToVerifyRules:
    return "fails to verify broadcasting rules";
  case utils::MatchBroadcastRuleResult::Success:
    return "";
  }
  llvm_unreachable("unhandled MatchBroadcastRuleResult case");
}

LogicalResult IdentityOp::verify() {
  utils::MatchBroadcastRuleResult res =
      utils::verifyTppIdentityBroadcastingRules(getInput().getType(),
                                                getOutput().getType());
  if (res != utils::MatchBroadcastRuleResult::Success)
    return emitOpError(getMatchBroadcastRuleMessage(res));
  return success();
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

static bool verifyMatmulShape(MemRefType memrefA, MemRefType memrefB,
                              MemRefType memrefC) {
  return !(memrefB.getRank() != 2 || memrefC.getRank() != 2 ||
           memrefA.getRank() != 2);
}

static bool verifyMatmulOperandsDims(ArrayRef<int64_t> shapeA,
                                     ArrayRef<int64_t> shapeB,
                                     ArrayRef<int64_t> shapeC) {
  int64_t m = shapeC[0];
  int64_t n = shapeC[1];
  int64_t k = shapeA[1];
  // Verify C(m, n) = A(m, k) B(k, n)
  return !(shapeB[0] != k || shapeB[1] != n || shapeA[0] != m);
}

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
  return !(memrefB.getRank() != 3 || memrefC.getRank() != 2 ||
           memrefA.getRank() != 3);
}

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

LogicalResult FusedBrgemmOp::verify() {
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

void FusedBrgemmOp::build(OpBuilder &builder, OperationState &state,
                          ValueRange inputs, Value output) {
  FusedBrgemmOp::build(builder, state, inputs[0], inputs[1], inputs[2], output);
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
  Type outputType = getOut().getType();
  if ((!lhsType.isa<ShapedType>()) || (!rhsType.isa<ShapedType>()) ||
      (!outputType.isa<ShapedType>()))
    return emitOpError("expects all operands to be shaped type");
  if (!utils::allOperandsHaveSameShapeAndStrides(
          {lhsType, rhsType, outputType}))
    return emitOpError(
        "requires all operands to have the same shape or strides");
  return success();
}

//===----------------------------------------------------------------------===//
// AddBCastOp
//===----------------------------------------------------------------------===//

// Accept only shaped operands for AddOp. We currently do not support
// broadcasting and TPP operations are memory to memory thus disallow scalar
// operand for now.
LogicalResult AddBCastOp::verify() {
  Type lhsType = getLhs().getType();
  Type rhsType = getRhs().getType();
  Type outputType = getOut().getType();
  if ((!lhsType.isa<ShapedType>()) || (!rhsType.isa<ShapedType>()) ||
      (!outputType.isa<ShapedType>()))
    return emitOpError("expects all operands to be shaped type");
  return success();
}
//===----------------------------------------------------------------------===//
// ReluOp
//===----------------------------------------------------------------------===//

LogicalResult ReluOp::verify() {
  Type inputType = getInput().getType();
  Type outputType = getOutput().getType();
  if ((!inputType.isa<ShapedType>()) || (!outputType.isa<ShapedType>()))
    return emitOpError("expects both operands to be shaped type");
  return success();
}

//===----------------------------------------------------------------------===//
// VNNIMatmulOp
//===----------------------------------------------------------------------===//

static bool verifyVNNIMatmulShape(MemRefType memrefA, MemRefType memrefB,
                                  MemRefType memrefC) {
  return !(memrefB.getRank() != 3 || memrefC.getRank() != 2 ||
           memrefA.getRank() != 2);
}

static bool verifyVNNIMatmulOperandsDims(ArrayRef<int64_t> shapeA,
                                         ArrayRef<int64_t> shapeB,
                                         ArrayRef<int64_t> shapeC) {
  int64_t m = shapeC[0];
  int64_t n = shapeC[1];
  int64_t k = shapeA[1];
  return !(shapeB[0] * shapeB[2] != k || shapeB[1] != n || shapeA[0] != m);
}

LogicalResult VNNIMatmulOp::verify() {
  MemRefType memrefA = getMatrixA().getType().cast<MemRefType>();
  MemRefType memrefB = getMatrixB().getType().cast<MemRefType>();
  MemRefType memrefC = getMatrixC().getType().cast<MemRefType>();
  assert(memrefB.getElementType().isBF16() && memrefB.getRank() == 3);
  if (!verifyVNNIMatmulShape(memrefA, memrefB, memrefC))
    return emitOpError("fails to verify operands shapes");
  if (!verifyVNNIMatmulOperandsDims(memrefA.getShape(), memrefB.getShape(),
                                    memrefC.getShape()))
    return emitOpError("fails to verify operands dimensions mismatch");
  return success();
}

void VNNIMatmulOp::build(OpBuilder &builder, OperationState &state,
                         ValueRange inputs, Value output) {
  VNNIMatmulOp::build(builder, state, inputs[0], inputs[1], output);
}

//===----------------------------------------------------------------------===//
// VNNIBrgemmOp
//===----------------------------------------------------------------------===//

static bool verifyVNNIBRGemmShape(MemRefType memrefA, MemRefType memrefB,
                                  MemRefType memrefC) {
  return !(memrefB.getRank() != 4 || memrefC.getRank() != 2 ||
           memrefA.getRank() != 3);
}

LogicalResult VNNIBrgemmOp::verify() {
  MemRefType tensorA = getBatchMatrixA().getType().cast<MemRefType>();
  MemRefType tensorB = getBatchMatrixB().getType().cast<MemRefType>();
  MemRefType matrixC = getMatrixC().getType().cast<MemRefType>();
  if (!verifyVNNIBRGemmShape(tensorA, tensorB, matrixC))
    return emitOpError("fails to verify operands shapes");
  // Check batch dimension.
  if (tensorB.getShape()[1] * tensorB.getShape()[3] != tensorA.getShape()[2])
    return emitOpError("fails to verify operands dimensions mismatch");
  return success();
}

void VNNIBrgemmOp::build(OpBuilder &builder, OperationState &state,
                         ValueRange inputs, Value output) {
  VNNIBrgemmOp::build(builder, state, inputs[0], inputs[1], output);
}

LogicalResult FusedVNNIBrgemmOp::verify() {
  MemRefType tensorA = getBatchMatrixA().getType().cast<MemRefType>();
  MemRefType tensorB = getBatchMatrixB().getType().cast<MemRefType>();
  MemRefType matrixC = getMatrixC().getType().cast<MemRefType>();
  if (!verifyVNNIBRGemmShape(tensorA, tensorB, matrixC))
    return emitOpError("fails to verify operands shapes");
  // Check batch dimension.
  if (tensorB.getShape()[1] * tensorB.getShape()[3] != tensorA.getShape()[2])
    return emitOpError("fails to verify operands dimensions mismatch");
  return success();
}

void FusedVNNIBrgemmOp::build(OpBuilder &builder, OperationState &state,
                              ValueRange inputs, Value output) {
  FusedVNNIBrgemmOp::build(builder, state, inputs[0], inputs[1], inputs[2],
                           output);
}
