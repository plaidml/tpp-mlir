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
// Utils
//===----------------------------------------------------------------------===//

static void printCommaSeparatedList(OpAsmPrinter &printer, ValueRange values) {
  printer << '(';
  for (auto [idx, value] : llvm::enumerate(values)) {
    printer << value << " : " << value.getType();
    if (idx != values.size() - 1)
      printer << ", ";
  }
  printer << ')';
}

static ParseResult parseTppOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  SmallVector<Type> operandsTypes;

  bool isMemRef = false;
  if (succeeded(parser.parseOptionalKeyword("ins")))
    isMemRef = true;

  // Parse operands.
  SmallVector<llvm::SMLoc> locsOperands;
  auto parseOperand = [&]() -> ParseResult {
    locsOperands.push_back(parser.getCurrentLocation());
    if (parser.parseOperand(operands.emplace_back()) ||
        parser.parseColonType(operandsTypes.emplace_back()))
      return failure();
    return success();
  };

  if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                     parseOperand)) {
    return failure();
  }

  if (isMemRef) {
    if (parser.parseKeyword("outs") || parser.parseLParen() ||
        parser.parseOperand(operands.emplace_back()) ||
        parser.parseColonType(operandsTypes.emplace_back()) ||
        parser.parseRParen())
      return failure();
  } else {
    // Parse result types.
    SmallVector<Type> resultTypes;
    llvm::SMLoc resultTypeLoc = parser.getCurrentLocation();
    if (parser.parseArrowTypeList(resultTypes) ||
        parser.addTypesToList(resultTypes, result.types))
      return failure();

    if (resultTypes.size() != 1) {
      return parser.emitError(resultTypeLoc,
                              "expect single result at tensor abstraction");
    }
  }

  // Validate operands.
  for (auto [idx, operand] : llvm::enumerate(operands)) {
    if (parser.resolveOperand(operand, operandsTypes[idx], result.operands))
      return failure();
    if (isMemRef && operandsTypes[idx].isa<RankedTensorType>())
      return parser.emitError(locsOperands[idx], "expect memref type");

    if (!isMemRef && operandsTypes[idx].isa<MemRefType>())
      return parser.emitError(locsOperands[idx], "expect tensor type");
  }

  NamedAttrList attrs;
  if (parser.parseOptionalAttrDictWithKeyword(attrs))
    return failure();
  result.addAttributes(attrs);
  return success();
}

// Print a tpp op. Note that `out` can be null. It is null for unary and binary
// at tensor abstraction. Ternary operations have `out` also at tensor
// abstraction.
void printTppOp(OpAsmPrinter &printer, ValueRange operands, Value out,
                TypeRange results, Operation *op, bool isTernary = false) {
  printer << ' ';
  if (results.empty()) {
    printer << "ins";
    printCommaSeparatedList(printer, operands);
    printer << ' ';
    printer << "outs";
    printCommaSeparatedList(printer, {out});
  } else {
    SmallVector<Value> tensorOperands = llvm::to_vector(operands);
    // Ternay op are += thus we need to pass `C` also at the tensor level.
    if (isTernary)
      tensorOperands.emplace_back(out);
    printCommaSeparatedList(printer, tensorOperands);
    printer << " -> (" << results << ")";
  }
  printer.printOptionalAttrDict((op)->getAttrs());
}

//===----------------------------------------------------------------------===//
// IdentityOp
//===----------------------------------------------------------------------===//

void IdentityOp::build(OpBuilder &builder, OperationState &result, Value input,
                       Value output) {
  return IdentityOp::build(builder, result, /*TypeRange=*/{}, input, output);
}

void IdentityOp::print(OpAsmPrinter &printer) {
  Value output = hasTensorSemantics() ? Value() : getOutput();
  printTppOp(printer, ValueRange{getInput()}, output, getResultTypes(), *this);
}

ParseResult IdentityOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTppOp(parser, result);
}

//===----------------------------------------------------------------------===//
// ReluOp
//===----------------------------------------------------------------------===//

void ReluOp::build(OpBuilder &builder, OperationState &result, Value input,
                   Value output) {
  return ReluOp::build(builder, result, /*TypeRange=*/{}, input, output);
}

void ReluOp::print(OpAsmPrinter &printer) {
  Value output = hasTensorSemantics() ? Value() : getOutput();
  printTppOp(printer, ValueRange{getInput()}, output, getResultTypes(), *this);
}

ParseResult ReluOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTppOp(parser, result);
}

//===----------------------------------------------------------------------===//
// AdddOp
//===----------------------------------------------------------------------===//

void AddOp::build(OpBuilder &builder, OperationState &result, Value lhs,
                  Value rhs, Value out) {
  return AddOp::build(builder, result, /*TypeRange=*/{}, lhs, rhs, out);
}

void AddOp::print(OpAsmPrinter &printer) {
  Value output = hasTensorSemantics() ? Value() : getOutput();
  printTppOp(printer, ValueRange{getLhs(), getRhs()}, output, getResultTypes(),
             *this);
}

ParseResult AddOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTppOp(parser, result);
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

static bool verifyMatmulShape(ShapedType shapedA, ShapedType shapedB,
                              ShapedType shapedC) {
  return !(shapedB.getRank() != 2 || shapedC.getRank() != 2 ||
           shapedA.getRank() != 2);
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
  auto shapedA = getAType();
  auto shapedB = getBType();
  auto shapedC = getCType();
  if (!verifyMatmulShape(shapedA, shapedB, shapedC))
    return emitOpError("fails to verify operands shapes");
  if (!verifyMatmulOperandsDims(shapedA.getShape(), shapedB.getShape(),
                                shapedC.getShape()))
    return emitOpError("fails to verify operands dimensions mismatch");
  return success();
}

void MatmulOp::build(OpBuilder &builder, OperationState &state,
                     ValueRange inputs, Value output) {
  MatmulOp::build(builder, state, /*TypeRange=*/{}, inputs[0], inputs[1],
                  output);
}

ParseResult MatmulOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTppOp(parser, result);
}

void MatmulOp::print(OpAsmPrinter &printer) {
  printTppOp(printer, ValueRange{getShapedA(), getShapedB()}, getShapedC(),
             getResultTypes(), *this, /*isTernary=*/true);
}

//===----------------------------------------------------------------------===//
// BrgemmOp
//===----------------------------------------------------------------------===//

static bool verifyBRGemmShape(ShapedType shapedA, ShapedType shapedB,
                              ShapedType shapedC) {
  return !(shapedB.getRank() != 3 || shapedC.getRank() != 2 ||
           shapedA.getRank() != 3);
}

LogicalResult BrgemmOp::verify() {
  auto shapedA = getShapedA().getType().cast<ShapedType>();
  auto shapedB = getShapedB().getType().cast<ShapedType>();
  auto shapedC = getShapedC().getType().cast<ShapedType>();
  if (!verifyBRGemmShape(shapedA, shapedB, shapedC))
    return emitOpError("fails to verify operands shapes");
  // Check batch dimension.
  if (shapedA.getShape()[0] != shapedB.getShape()[0])
    return emitOpError("fails to verify operands dimensions mismatch");
  // Check all others that must be 'matmul' like.
  if (!verifyMatmulOperandsDims(shapedA.getShape().drop_front(),
                                shapedB.getShape().drop_front(),
                                shapedC.getShape()))
    return emitOpError("fails to verify operands dimensions mismatch");
  return success();
}

void BrgemmOp::build(OpBuilder &builder, OperationState &state,
                     ValueRange inputs, Value output) {
  BrgemmOp::build(builder, state, /*TypeRange=*/{}, inputs[0], inputs[1],
                  output);
}

ParseResult BrgemmOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTppOp(parser, result);
}

void BrgemmOp::print(OpAsmPrinter &printer) {
  printTppOp(printer, ValueRange{getShapedA(), getShapedB()}, getShapedC(),
             getResultTypes(), *this, /*isTernary=*/true);
}

//===----------------------------------------------------------------------===//
// FusedBrgemmOp
//===----------------------------------------------------------------------===//

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
  FusedBrgemmOp::build(
      builder, state, inputs[0], inputs[1], inputs[2],
      tpp::FusedOpTypeAttr::get(builder.getContext(), tpp::FusedOpType::NONE),
      output);
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

//===----------------------------------------------------------------------===//
// FusedVNNIBrgemmOp
//===----------------------------------------------------------------------===//

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
