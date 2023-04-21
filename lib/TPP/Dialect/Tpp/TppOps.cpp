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
  int numberOfInputs = operands.size();
  int numberOfOutputs = 0;

  if (isMemRef) {
    locsOperands.push_back(parser.getCurrentLocation());
    if (parser.parseKeyword("outs") || parser.parseLParen() ||
        parser.parseOperand(operands.emplace_back()) ||
        parser.parseColonType(operandsTypes.emplace_back()) ||
        parser.parseRParen())
      return failure();
    numberOfOutputs = operands.size() - numberOfInputs;
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

  // Validate operands. Scan each operand one-by-one to emit
  // better diagnostic.
  for (auto [idx, operand] : llvm::enumerate(operands)) {
    if (parser.resolveOperand(operand, operandsTypes[idx], result.operands))
      return failure();
    if (isMemRef && operandsTypes[idx].isa<RankedTensorType>())
      return parser.emitError(locsOperands[idx], "expect memref type");
    if (!isMemRef && operandsTypes[idx].isa<MemRefType>())
      return parser.emitError(locsOperands[idx], "expect tensor type");
  }

  NamedAttrList attrs;
  if (parser.parseOptionalAttrDict(attrs))
    return failure();
  // Check if we parsed `operand_segment_sizes` already, otherwise add it.
  if (!attrs.get("operand_segment_sizes")) {
    auto operandSegmentSize = parser.getBuilder().getDenseI32ArrayAttr(
        {numberOfInputs, numberOfOutputs});
    result.addAttribute("operand_segment_sizes", operandSegmentSize);
  }
  result.addAttributes(attrs);
  return success();
}

// Print a tpp op. Note that `out` can be null. It is null for unary and binary
// at tensor abstraction.
static void printTppOp(OpAsmPrinter &printer, ValueRange operands,
                       ValueRange outs, TypeRange results, Operation *op) {
  printer << ' ';
  if (results.empty()) {
    printer << "ins";
    printCommaSeparatedList(printer, operands);
    printer << ' ';
    printer << "outs";
    printCommaSeparatedList(printer, outs);
  } else {
    printCommaSeparatedList(printer, operands);
    printer << " -> (" << results << ")";
  }
  printer.printOptionalAttrDict(op->getAttrs(),
                                /*elidedAttrs=*/{"operand_segment_sizes"});
}

static void tppOpBuilder(OpBuilder &builder, OperationState &state,
                         ValueRange inputs, ValueRange outputs) {
  assert(outputs.size() >= 1);
  state.addOperands(inputs);
  if (auto rankedOutput =
          outputs[0].getType().dyn_cast_or_null<RankedTensorType>()) {
    state.addTypes(outputs.getTypes());
    state.addAttribute(
        "operand_segment_sizes",
        builder.getDenseI32ArrayAttr(
            {static_cast<int>(inputs.size()), /*numOutputs=*/0}));
  } else {
    state.addOperands(outputs);
    state.addAttribute(
        "operand_segment_sizes",
        builder.getDenseI32ArrayAttr({static_cast<int>(inputs.size()),
                                      static_cast<int>(outputs.size())}));
  }
}

static void getEffectsImpl(
    TppOp tppOp,
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (tppOp.hasTensorSemantics())
    return;
  for (auto operand : tppOp.getInputs()) {
    if (!operand.getType().isa<MemRefType>())
      continue;
    effects.emplace_back(MemoryEffects::Read::get(), operand,
                         SideEffects::DefaultResource::get());
  }
  effects.emplace_back(MemoryEffects::Write::get(), tppOp.getOutput(),
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// IdentityOp
//===----------------------------------------------------------------------===//

void IdentityOp::build(OpBuilder &builder, OperationState &state, Value input,
                       Value output) {
  tppOpBuilder(builder, state, input, output);
}

void IdentityOp::print(OpAsmPrinter &printer) {
  printTppOp(printer, getInputs(), getOutputs(), getResultTypes(), *this);
}

ParseResult IdentityOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTppOp(parser, result);
}

void IdentityOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getEffectsImpl(*this, effects);
}

//===----------------------------------------------------------------------===//
// ReluOp
//===----------------------------------------------------------------------===//

void ReluOp::build(OpBuilder &builder, OperationState &state, Value input,
                   Value output) {
  tppOpBuilder(builder, state, input, output);
}

void ReluOp::print(OpAsmPrinter &printer) {
  printTppOp(printer, getInputs(), getOutputs(), getResultTypes(), *this);
}

ParseResult ReluOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTppOp(parser, result);
}

void ReluOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getEffectsImpl(*this, effects);
}

//===----------------------------------------------------------------------===//
// ZeroOp
//===----------------------------------------------------------------------===//

void ZeroOp::build(OpBuilder &builder, OperationState &state, Value input,
                   Value output) {
  tppOpBuilder(builder, state, input, output);
}

void ZeroOp::print(OpAsmPrinter &printer) {
  printTppOp(printer, getInputs(), getOutputs(), getResultTypes(), *this);
}

ParseResult ZeroOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTppOp(parser, result);
}

LogicalResult ZeroOp::verify() {
  // At tensor abstraction computation result is always placed in a new tensor
  // so skip validation.
  if (hasTensorSemantics())
    return success();

  auto input = getInputs()[0];
  auto output = getOutputs()[0];

  if (input != output)
    return emitOpError("fails to verify in-place computation");

  return success();
}

void ZeroOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getEffectsImpl(*this, effects);
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

void AddOp::build(OpBuilder &builder, OperationState &state, ValueRange inputs,
                  Value output) {
  tppOpBuilder(builder, state, inputs, output);
}

void AddOp::print(OpAsmPrinter &printer) {
  printTppOp(printer, getInputs(), getOutputs(), getResultTypes(), *this);
}

ParseResult AddOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTppOp(parser, result);
}

void AddOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getEffectsImpl(*this, effects);
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
  auto shapedA = getInputs()[0].getType().cast<ShapedType>();
  auto shapedB = getInputs()[1].getType().cast<ShapedType>();
  auto shapedC = (hasTensorSemantics()) ? getResultType().cast<ShapedType>()
                                        : getOutputType().cast<ShapedType>();
  if (!verifyMatmulShape(shapedA, shapedB, shapedC))
    return emitOpError("fails to verify operands shapes");
  if (!verifyMatmulOperandsDims(shapedA.getShape(), shapedB.getShape(),
                                shapedC.getShape()))
    return emitOpError("fails to verify operands dimensions mismatch");
  return success();
}

void MatmulOp::build(OpBuilder &builder, OperationState &state,
                     ValueRange inputs, Value output) {
  tppOpBuilder(builder, state, inputs, output);
}

ParseResult MatmulOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTppOp(parser, result);
}

void MatmulOp::print(OpAsmPrinter &printer) {
  printTppOp(printer, getInputs(), getOutputs(), getResultTypes(), *this);
}

void MatmulOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getEffectsImpl(*this, effects);
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
  auto shapedA = getInputs()[0].getType().cast<ShapedType>();
  auto shapedB = getInputs()[1].getType().cast<ShapedType>();
  auto shapedC = (hasTensorSemantics()) ? getResultType().cast<ShapedType>()
                                        : getOutputType().cast<ShapedType>();
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
  tppOpBuilder(builder, state, inputs, output);
}

ParseResult BrgemmOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTppOp(parser, result);
}

void BrgemmOp::print(OpAsmPrinter &printer) {
  printTppOp(printer, getInputs(), getOutputs(), getResultTypes(), *this);
}

void BrgemmOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getEffectsImpl(*this, effects);
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
