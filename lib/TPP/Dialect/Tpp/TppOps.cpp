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
#include "TPP/VNNIUtils.h"
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
  printer.printOptionalAttrDict(
      op->getAttrs(),
      /*elidedAttrs=*/{"operand_segment_sizes", "unary_kind", "binary_kind"});
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
// GemmOp
//===----------------------------------------------------------------------===//

template <typename OpTy>
static LogicalResult
validateVnniGemmOperand(OpTy operation, ArrayRef<int64_t> shape,
                        Type elementType, int dimI, int dimJ) {

  if (shape.size() != 2 && shape.size() != 3) {
    return operation->emitOpError("expect rank 2 or 3 for operand 1");
  }
  if (shape.size() == 2) {
    if (shape[0] != dimI || shape[1] != dimJ)
      return operation->emitOpError("operand 1 fails to verify expected shape");
    return success();
  }
  if (!elementType.isBF16()) {
    return operation->emitOpError() << "operand 1 invalid element type for "
                                       "VNNI layout expect bf16, but got: "
                                    << elementType << "\n";
  }
  if (shape[2] != vnni::utils::getVnniBlockingFactor(elementType)) {
    return operation->emitOpError() << "operand 1 invalid VNNI layout expect "
                                       "inner dims to be 2 or 4, but got: "
                                    << shape[shape.size() - 1] << "\n";
  }
  // VNNI layout: [K/VNNI][J][VNNI]
  if (shape[0] * shape[2] != dimI || shape[1] != dimJ)
    return operation->emitOpError("operand 1 fails to verify expected shape");
  return success();
}

template <typename OpTy>
static LogicalResult verifyGemmLikeOperands(OpTy operation) {

  static_assert(llvm::is_one_of<OpTy, BrgemmOp, FusedBrgemmOp, GemmOp>::value,
                "applies to brgemm, fused_brgemm or gemm operations");

  auto shapedA = operation.getInputs()[0].getType().template cast<ShapedType>();
  auto shapedB = operation.getInputs()[1].getType().template cast<ShapedType>();
  auto shapedC = operation.getInputs()[2].getType().template cast<ShapedType>();
  auto shapedResult =
      (operation.hasTensorSemantics())
          ? operation.getResultType().template cast<ShapedType>()
          : operation.getOutputType().template cast<ShapedType>();

  if (shapedC != shapedResult) {
    return operation.emitOpError()
           << "result type differs from destination operand type";
  }

  // Validate operand C.
  if (shapedC.getRank() != 2) {
    return operation.emitOpError()
           << "operand 2 expects rank 2, but got: " << shapedC.getRank()
           << "\n";
  }
  int64_t m = shapedC.getShape()[0];
  int64_t n = shapedC.getShape()[1];

  // Validate operand A.
  bool isGemmOp = isa<tpp::GemmOp>(operation.getOperation());
  // Brgemm has size 3 for A while Gemm has size 2.
  auto expectRankA = (isGemmOp) ? 2 : 3;
  if (shapedA.getRank() != expectRankA) {
    return operation.emitOpError()
           << "operand 0 expects rank " << expectRankA
           << ", but got: " << shapedA.getRank() << "\n";
  }
  int64_t k = shapedA.getShape()[shapedA.getRank() - 1];
  // On A operand the 'm' dimension is at position 0 for Gemm while 1 for
  // Brgemm.
  int64_t idxMOnA = (isGemmOp) ? 0 : 1;
  if (shapedA.getShape()[idxMOnA] != m)
    return operation.emitOpError("operand 0 fails to verify expected shape");

  // Validate operand B.
  int64_t batch = shapedA.getShape()[0];
  if (!isGemmOp && shapedB.getShape()[0] != batch)
    return operation.emitOpError("operand 1 fails to verify expected shape");
  // Drop the batch dim for brgemm, already checked.
  auto shapeB =
      (isGemmOp) ? shapedB.getShape() : shapedB.getShape().drop_front();
  return validateVnniGemmOperand(operation, shapeB, shapedB.getElementType(), k,
                                 n);
}

// Verify gemm operation.
LogicalResult GemmOp::verify() { return verifyGemmLikeOperands(*this); }

void GemmOp::build(OpBuilder &builder, OperationState &state, ValueRange inputs,
                   Value output) {
  tppOpBuilder(builder, state, inputs, output);
}

ParseResult GemmOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTppOp(parser, result);
}

void GemmOp::print(OpAsmPrinter &printer) {
  printTppOp(printer, getInputs(), getOutputs(), getResultTypes(), *this);
}

void GemmOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getEffectsImpl(*this, effects);
}

//===----------------------------------------------------------------------===//
// BrgemmOp
//===----------------------------------------------------------------------===//

LogicalResult BrgemmOp::verify() { return verifyGemmLikeOperands(*this); }

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

LogicalResult FusedBrgemmOp::verify() { return verifyGemmLikeOperands(*this); }

void FusedBrgemmOp::build(OpBuilder &builder, OperationState &state,
                          ValueRange inputs, Value output,
                          FusedUnaryOpKindAttr unaryKind,
                          FusedBinaryOpKindAttr binaryKind) {
  tppOpBuilder(builder, state, inputs, output);
  state.addAttribute("unary_kind", unaryKind);
  state.addAttribute("binary_kind", binaryKind);
}

template <typename EnumClass>
static ParseResult parseEnum(EnumClass &value, OpAsmParser &parser) {
  StringRef flag;
  auto loc = parser.getCurrentLocation();
  if (parser.parseKeyword(&flag))
    return failure();
  auto flagAttr = symbolizeEnum<EnumClass>(flag);
  if (!flagAttr)
    return parser.emitError(loc, "invalid enum ") << flag;
  value = *flagAttr;
  return success();
}

ParseResult FusedBrgemmOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parser.parseLSquare() || parser.parseKeyword("unary") ||
      parser.parseEqual())
    return failure();
  FusedUnaryOpKind unaryKind;
  if (parseEnum(unaryKind, parser))
    return failure();
  if (parser.parseComma() || parser.parseKeyword("binary") ||
      parser.parseEqual())
    return failure();
  FusedBinaryOpKind binaryKind;
  if (parseEnum(binaryKind, parser))
    return failure();
  if (parser.parseRSquare())
    return failure();
  auto ctx = parser.getBuilder().getContext();
  result.addAttribute("unary_kind", FusedUnaryOpKindAttr::get(ctx, unaryKind));
  result.addAttribute("binary_kind",
                      FusedBinaryOpKindAttr::get(ctx, binaryKind));
  return parseTppOp(parser, result);
}

void FusedBrgemmOp::print(OpAsmPrinter &printer) {
  printer << " [unary = " << tpp::stringifyFusedUnaryOpKind(getUnaryKind())
          << ", binary = " << tpp::stringifyFusedBinaryOpKind(getBinaryKind())
          << "]";
  printTppOp(printer, getInputs(), getOutputs(), getResultTypes(), *this);
}

void FusedBrgemmOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getEffectsImpl(*this, effects);
}
