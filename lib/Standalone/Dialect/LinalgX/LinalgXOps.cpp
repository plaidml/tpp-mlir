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

using namespace mlir;
using namespace mlir::linalgx;

ArrayAttr ToBlockLayout::getIndexingMaps() {
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

ArrayAttr ToBlockLayout::iterator_types() {
  int64_t numLoops = getTiedIndexingMap(getInputOperand(0)).getNumDims();
  return Builder(getContext())
      .getStrArrayAttr(
          SmallVector<StringRef, 8>(numLoops, getParallelIteratorTypeName()));
}
/*
// `LinalgOps.cpp`
/// Common parsing used for both named structured ops created by ods-gen and by
/// manually defined C++ ops. Does not handle regions.
static ParseResult
parseCommonStructuredOpParts(OpAsmParser &parser, OperationState &result,
                             SmallVectorImpl<Type> &inputTypes,
                             SmallVectorImpl<Type> &outputTypes) {
  SMLoc inputsOperandsLoc, outputsOperandsLoc;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> inputsOperands,
      outputsOperands;

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (succeeded(parser.parseOptionalKeyword("ins"))) {
    if (parser.parseLParen())
      return failure();

    inputsOperandsLoc = parser.getCurrentLocation();
    if (parser.parseOperandList(inputsOperands) ||
        parser.parseColonTypeList(inputTypes) || parser.parseRParen())
      return failure();
  }

  if (succeeded(parser.parseOptionalKeyword("outs"))) {
    outputsOperandsLoc = parser.getCurrentLocation();
    if (parser.parseLParen() || parser.parseOperandList(outputsOperands) ||
        parser.parseColonTypeList(outputTypes) || parser.parseRParen())
      return failure();
  }

  if (parser.resolveOperands(inputsOperands, inputTypes, inputsOperandsLoc,
                             result.operands) ||
      parser.resolveOperands(outputsOperands, outputTypes, outputsOperandsLoc,
                             result.operands))
    return failure();

  result.addAttribute("operand_segment_sizes",
                      parser.getBuilder().getI32VectorAttr(
                          {static_cast<int32_t>(inputsOperands.size()),
                           static_cast<int32_t>(outputsOperands.size())}));
  return success();
}

// see `LinalgOps.cpp`
static void printCommonStructuredOpParts(OpAsmPrinter &p, ValueRange inputs,
                                         ValueRange outputs) {
  if (!inputs.empty())
    p << " ins(" << inputs << " : " << inputs.getTypes() << ")";
  if (!outputs.empty())
    p << " outs(" << outputs << " : " << outputs.getTypes() << ")";
}

// see `LinalgOps.cpp`
using RegionBuilderFn = llvm::function_ref<void(ImplicitLocOpBuilder &, Block &,
                                                 ArrayRef<NamedAttribute>)>;
static ParseResult parseNamedStructuredOp(OpAsmParser &parser,
                                          OperationState &result,
                                          unsigned numRegionArgs,
                                          RegionBuilderFn regionBuilder) {
  // TODO: Enable when ods-gen supports captures.
  SmallVector<Type, 1> inputTypes, outputTypes;
  if (parseCommonStructuredOpParts(parser, result, inputTypes, outputTypes))
    return failure();

  // TODO: consider merging results parsing into region parsing.
  // Need to wait for declarative assembly resolution to decide.
  SmallVector<Type, 1> outputTensorsTypes;
  if (parseNamedStructuredOpResults(parser, outputTensorsTypes))
    return failure();
  result.addTypes(outputTensorsTypes);

  std::unique_ptr<Region> region = std::make_unique<Region>();
  if (parseNamedStructuredOpRegion(parser, *region, numRegionArgs, inputTypes,
                                   outputTypes, result.attributes.getAttrs(),
                                   regionBuilder))
    return failure();
  result.addRegion(std::move(region));

  return success();
}

ParseResult ToBlockLayout::parse(OpAsmParser &parser, OperationState &result) {
  parseNamedStructuredOp(parser, result, ToBlockLayout.getNumRegionArgs(),
                         ToBlockLayout::getRegionBuilder());
}

void ToBlockLayout::print(OpAsmPrinter &p) {}
*/
std::string ToBlockLayout::getLibraryCallName() { return "to_block_layout"; }

void ToBlockLayout::regionBuilder(ImplicitLocOpBuilder &b, Block &block,
                                  llvm::ArrayRef<NamedAttribute> attrs = {}) {
  assert(block.getNumArguments() == 2 && "CopyOp regionBuilder expects 2 args");
  b.create<linalg::YieldOp>(block.getArgument(0));
}

#define GET_OP_CLASSES
#include "Standalone/Dialect/LinalgX/LinalgXOps.cpp.inc"
