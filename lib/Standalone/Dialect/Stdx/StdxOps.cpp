//===- StdxOps.cpp - Stdx dialect ops ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/Stdx/StdxOps.h"
#include "Standalone/Dialect/Stdx/StdxDialect.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::stdx;

Region &ClosureOp::getLoopBody() { return getRegion(); }

void ClosureOp::build(OpBuilder &builder, OperationState &result, Value out,
                      ValueRange inits) {
  result.addOperands(inits);
  result.addOperands(out);
  result.addTypes(out.getType());

  Region *bodyRegion = result.addRegion();
  bodyRegion->push_back(new Block);
  Block &bodyBlock = bodyRegion->front();
  for (Value v : inits)
    bodyBlock.addArgument(v.getType(), v.getLoc());
  bodyBlock.addArgument(out.getType(), out.getLoc());
  ClosureOp::ensureTerminator(*bodyRegion, builder, result.location);
}

// Prints the initialization list in the form of
//   <prefix>(%inner = %outer, %inner2 = %outer2, <...>)
// where 'inner' values are assumed to be region arguments and 'outer' values
// are regular SSA values.
static void printInitializationList(OpAsmPrinter &p,
                                    Block::BlockArgListType blocksArgs,
                                    ValueRange initializers,
                                    StringRef prefix = "") {
  assert(blocksArgs.size() == initializers.size() &&
         "expected same length of arguments and initializers");
  if (initializers.empty())
    return;

  p << prefix << '(';
  llvm::interleaveComma(llvm::zip(blocksArgs, initializers), p, [&](auto it) {
    p << std::get<0>(it) << " = " << std::get<1>(it);
  });
  p << ")";
}

void ClosureOp::print(OpAsmPrinter &p) {
  printInitializationList(p, getRegionIterArgs(), getIterOperands(),
                          " init_args");
  if (!getIterOperands().empty())
    p << " -> (" << getIterOperands().getTypes() << ')';
  p << ' ';
  p.printRegion(getRegion(),
                /*printEntryBlockArgs=*/true,
                /*printBlockTerminators=*/hasIterOperands());
  p.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult ClosureOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::Argument, 4> regionArgs;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;

  if (parser.parseKeyword("init_args"))
    return failure();

  // Parse assignment list and type list for input (init_args).
  SmallVector<Type> argumentsTypes;
  if (parser.parseAssignmentList(regionArgs, operands) ||
      parser.parseArrowTypeList(argumentsTypes))
    return failure();

  if (parser.parseKeyword("init_outs"))
    return failure();

  // Parse assignment list for result (init_outs).
  if (parser.parseAssignmentList(regionArgs, operands) ||
      parser.parseArrowTypeList(result.types))
    return failure();
  llvm::append_range(argumentsTypes, result.types);

  assert(argumentsTypes.size() == regionArgs.size() &&
         "expected same length of arguments");
  assert(argumentsTypes.size() == operands.size() &&
         "expected same length of arguments");

  for (auto argOperandType : llvm::zip(regionArgs, operands, argumentsTypes)) {
    Type type = std::get<2>(argOperandType);
    std::get<0>(argOperandType).type = type;
    if (parser.resolveOperand(std::get<1>(argOperandType), type,
                              result.operands))
      return failure();
  }

  // Parse the body region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs))
    return failure();

  ClosureOp::ensureTerminator(*body, parser.getBuilder(), result.location);

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

#define GET_OP_CLASSES
#include "Standalone/Dialect/Stdx/StdxOps.cpp.inc"
