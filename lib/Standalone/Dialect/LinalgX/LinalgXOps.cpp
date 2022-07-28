//===- LinalgXOps.cpp - LinalgX dialect ops ---------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/LinalgX/LinalgXOps.h"
#include "Standalone/Dialect/LinalgX/LinalgXDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser/Parser.h"

using namespace mlir;
using namespace mlir::linalgx;
using namespace mlir::bufferization;

using RegionBuilderFn = llvm::function_ref<void(ImplicitLocOpBuilder &, Block &,
                                                ArrayRef<NamedAttribute>)>;
// taken from linalgOps.cpp
/// Fills the region of a structured operation using the provided
/// `regionBuilder`. The method is used by both named structured ops created by
/// ods-gen and by manually defined C++ ops. It is called by both builders and
/// parsers and creates a block with arguments corresponding to the elemental
/// types of `inputTypes` and `outputTypes`. All output types are asserted to be
/// ShapedType.
static void fillStructuredOpRegion(OpBuilder &opBuilder, Region &region,
                                   TypeRange inputTypes, TypeRange outputTypes,
                                   ArrayRef<NamedAttribute> attrs,
                                   RegionBuilderFn regionBuilder) {
  assert(llvm::all_of(outputTypes, [](Type t) { return t.isa<ShapedType>(); }));

  // TODO: atm all operands go through getElementTypeOrSelf,
  // reconsider when we have evidence we need to.
  SmallVector<Type, 8> argTypes;
  SmallVector<Location, 8> argLocs;
  for (auto containers : {inputTypes, outputTypes}) {
    for (auto t : containers) {
      argTypes.push_back(getElementTypeOrSelf(t));

      // TODO: Pass in a proper location here.
      argLocs.push_back(opBuilder.getUnknownLoc());
    }
  }

  // RAII.
  OpBuilder::InsertionGuard guard(opBuilder);
  Block *body =
      opBuilder.createBlock(&region, /*insertPt=*/{}, argTypes, argLocs);

  opBuilder.setInsertionPointToStart(body);
  ImplicitLocOpBuilder b(opBuilder.getUnknownLoc(), opBuilder);
  regionBuilder(b, *body, attrs);

  // indexing_maps is an auto-generated method.

  // iterator_types is an auto-generated method.
}

// taken from LinalgOps.cpp
/// Creates a structured operation given `inputs`, `outputs`, `inputMap`,
/// `outputMap` and `attributes`. The result types are derived automatically if
/// `resultTensorTypes` is none. The body of the operation is filled using
/// `regionBuilder`. All ods-gen created structured operations use the method to
/// implement their builders.
static void buildStructuredOp(OpBuilder &b, OperationState &state,
                              llvm::Optional<TypeRange> resultTensorTypes,
                              ValueRange inputs, ValueRange outputs,
                              AffineMap inputMap, AffineMap outputMap,
                              ArrayRef<NamedAttribute> attributes,
                              RegionBuilderFn regionBuilder) {
  // Derive the result types if needed.
  SmallVector<Type> derivedResultTypes =
      resultTensorTypes.value_or(TypeRange());
  if (!resultTensorTypes)
    copy_if(outputs.getTypes(), std::back_inserter(derivedResultTypes),
            [](Type type) { return type.isa<RankedTensorType>(); });

  state.addOperands(inputs);
  state.addOperands(outputs);
  state.addTypes(derivedResultTypes);
  state.addAttributes(attributes);
  // These are our extensions
  state.addAttribute("inputMap", AffineMapAttr::get(inputMap));
  state.addAttribute("outputMap", AffineMapAttr::get(outputMap));
  // end our extensions
  state.addAttribute(
      "operand_segment_sizes",
      b.getI32VectorAttr({static_cast<int32_t>(inputs.size()),
                          static_cast<int32_t>(outputs.size())}));

  // Create and fill the region of the structured operation.
  Region &region = *state.addRegion();
  fillStructuredOpRegion(b, region, TypeRange(inputs), TypeRange(outputs),
                         state.attributes.getAttrs(), regionBuilder);
}

//===----------------------------------------------------------------------===//
// Relayout
//===----------------------------------------------------------------------===//

ArrayAttr Relayout::getIndexingMaps() {
  MLIRContext *context = getContext();
  auto maybeInputMap = getInputMap();
  auto maybeOutputMap = getOutputMap();
  int64_t inputRank = getRank(getInputOperand(0));
  int64_t outputRank = getRank(getOutputOperand(0));
  return Builder(getContext())
      .getAffineMapArrayAttr(
          {linalg::extractOrIdentityMap(maybeInputMap, inputRank, context),
           linalg::extractOrIdentityMap(maybeOutputMap, outputRank, context)});
}

ArrayAttr Relayout::iterator_types() {
  int64_t numLoops = getTiedIndexingMap(getInputOperand(0)).getNumDims();
  return Builder(getContext())
      .getStrArrayAttr(
          SmallVector<StringRef, 8>(numLoops, getParallelIteratorTypeName()));
}

std::string Relayout::getLibraryCallName() {
  return "relayout_to_block_layout_and_back";
}

void Relayout::regionBuilder(ImplicitLocOpBuilder &b, Block &block,
                             llvm::ArrayRef<NamedAttribute> attrs = {}) {
  assert(block.getNumArguments() == 2 && "CopyOp regionBuilder expects 2 args");
  b.create<linalg::YieldOp>(block.getArgument(0));
}

static void getGenericEffectsImpl(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects,
    ValueRange results, ValueRange inputBuffers, ValueRange outputs) {
  for (Value value : inputBuffers)
    effects.emplace_back(MemoryEffects::Read::get(), value,
                         SideEffects::DefaultResource::get());

  for (Value value : outputs)
    effects.emplace_back(MemoryEffects::Write::get(), value,
                         SideEffects::DefaultResource::get());
}

void Relayout::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  SmallVector<Value> inputBuffers = getInputBufferOperands();
  SmallVector<Value> outputBuffers = getOutputBufferOperands();
  getGenericEffectsImpl(effects, getOperation()->getResults(), inputBuffers,
                        outputBuffers);
}

static ParseResult
parseInputOutputAndMaps(OpAsmParser &parser, OperationState &result,
                        Type &inputType, AffineMapAttr &inputMap,
                        Type &outputType, AffineMapAttr &outputMap) {
  OpAsmParser::UnresolvedOperand inputOperand, outputOperand;
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (succeeded(parser.parseKeyword("ins")))
    if (parser.parseLParen())
      return failure();

  if (parser.parseOperand(inputOperand) || parser.parseColonType(inputType) ||
      parser.parseComma())
    return failure();

  if (parser.parseCustomAttributeWithFallback(inputMap, Type(), "inputMap",
                                              result.attributes) ||
      parser.parseRParen())
    return failure();

  if (succeeded(parser.parseKeyword("outs")))
    if (parser.parseLParen())
      return failure();

  if (parser.parseOperand(outputOperand) || parser.parseColonType(outputType) ||
      parser.parseComma())
    return failure();

  if (parser.parseCustomAttributeWithFallback(outputMap, Type(), "outputMap",
                                              result.attributes) ||
      parser.parseRParen())
    return failure();

  if (parser.resolveOperand(inputOperand, inputType, result.operands) ||
      parser.resolveOperand(outputOperand, outputType, result.operands))
    return failure();

  result.addAttribute("operand_segment_sizes",
                      parser.getBuilder().getI32VectorAttr({1, 1}));

  return success();
}

ParseResult Relayout::parse(OpAsmParser &parser, OperationState &state) {
  Type inputType, outputType;
  AffineMapAttr inputMap, outputMap;
  if (parseInputOutputAndMaps(parser, state, inputType, inputMap, outputType,
                              outputMap))
    return failure();

  if (parser.parseOptionalArrowTypeList(state.types))
    return failure();

  std::unique_ptr<Region> region = std::make_unique<Region>();
  OpBuilder opBuilder(parser.getContext());
  fillStructuredOpRegion(opBuilder, *region, TypeRange(inputType),
                         TypeRange(outputType), state.attributes.getAttrs(),
                         regionBuilder);
  state.addRegion(std::move(region));
  return success();
}

static void printResult(OpAsmPrinter &printer, TypeRange resultTypes) {
  if (resultTypes.empty())
    return printer.printOptionalArrowTypeList(resultTypes);
}

static void printOperands(OpAsmPrinter &printer, ValueRange inputs,
                          ValueRange outputs, AffineMap inputMap,
                          AffineMap outputMap) {
  assert(inputs.size() == 1 && "expect single input");
  assert(outputs.size() == 1 && "expect single output");
  Value input = inputs[0];
  Value output = outputs[0];
  printer << " ins(" << input << " : " << input.getType() << ", " << inputMap
          << ")";
  printer << " outs(" << output << " : " << output.getType() << ", "
          << outputMap << ")";
}

void Relayout::print(OpAsmPrinter &printer) {
  // print operands.
  printOperands(printer, inputs(), outputs(), getInputMap(), getOutputMap());
  // print results.
  printResult(printer, this->getResultTypes());
  // Region is elided.
}

#define GET_OP_CLASSES
#include "Standalone/Dialect/LinalgX/LinalgXOps.cpp.inc"
