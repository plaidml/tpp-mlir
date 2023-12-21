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

namespace {
constexpr std::string_view INS = "ins";
constexpr std::string_view OUTS = "outs";
constexpr std::string_view OPERAND_SEGMENT_SIZE = "operandSegmentSizes";
constexpr std::string_view UNARY_KIND = "unary_kind";
constexpr std::string_view BINARY_KIND = "binary_kind";
} // namespace

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
  if (succeeded(parser.parseOptionalKeyword(INS)))
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
    if (parser.parseKeyword(OUTS) || parser.parseLParen() ||
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
  // Check if we parsed `operandSegmentSizes` already, otherwise add it.
  if (!attrs.get(OPERAND_SEGMENT_SIZE)) {
    auto operandSegmentSize = parser.getBuilder().getDenseI32ArrayAttr(
        {numberOfInputs, numberOfOutputs});
    result.addAttribute(OPERAND_SEGMENT_SIZE, operandSegmentSize);
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
    printer << INS;
    printCommaSeparatedList(printer, operands);
    printer << ' ';
    printer << OUTS;
    printCommaSeparatedList(printer, outs);
  } else {
    printCommaSeparatedList(printer, operands);
    printer << " -> (" << results << ")";
  }
  printer.printOptionalAttrDict(
      op->getAttrs(),
      /*elidedAttrs=*/{OPERAND_SEGMENT_SIZE, UNARY_KIND, BINARY_KIND});
}

static void tppOpBuilderMemRef(OpBuilder &builder, OperationState &state,
                               ValueRange inputs, Value output) {
  assert(output.getType().isa<MemRefType>());
  state.addOperands(inputs);
  state.addOperands(output);
  state.addAttribute(OPERAND_SEGMENT_SIZE, builder.getDenseI32ArrayAttr(
                                               {static_cast<int>(inputs.size()),
                                                /*numOutputs=*/1}));
}

static void tppOpBuilderTensor(OpBuilder &builder, OperationState &state,
                               ValueRange inputs, Type outputType) {
  assert(outputType.isa<RankedTensorType>());
  state.addOperands(inputs);
  state.addTypes(outputType);
  state.addAttribute(OPERAND_SEGMENT_SIZE,
                     builder.getDenseI32ArrayAttr(
                         {static_cast<int>(inputs.size()), /*numOutputs=*/0}));
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
  tppOpBuilderMemRef(builder, state, input, output);
}

void IdentityOp::build(OpBuilder &builder, OperationState &state, Value input,
                       Type outputType) {
  tppOpBuilderTensor(builder, state, input, outputType);
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

// Builder for memref abstraction.
void ReluOp::build(OpBuilder &builder, OperationState &state, Value input,
                   Value output) {
  tppOpBuilderMemRef(builder, state, input, output);
}

// Builder for tensor abstraction.
void ReluOp::build(OpBuilder &builder, OperationState &state, Value input,
                   Type outputType) {
  tppOpBuilderTensor(builder, state, input, outputType);
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

// Builder for memref abstraction.
void ZeroOp::build(OpBuilder &builder, OperationState &state, Value input,
                   Value output) {
  tppOpBuilderMemRef(builder, state, input, output);
}

// Builder for tensor abstraction.
void ZeroOp::build(OpBuilder &builder, OperationState &state, Value input,
                   Type outputType) {
  tppOpBuilderTensor(builder, state, input, outputType);
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
