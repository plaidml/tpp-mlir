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

ParseResult ClosureOp::parse(OpAsmParser &parser, OperationState &result) {
  Builder &builder = parser.getBuilder();

  SmallVector<OpAsmParser::Argument> entryArgs;
  SmallVector<DictionaryAttr> resultAttrs;
  SmallVector<Type> resultTypes;
  bool isVariadic;

  if (failed(function_interface_impl::parseFunctionSignature(
          parser, /*allowVariadic=*/false, entryArgs, isVariadic, resultTypes,
          resultAttrs)))
    return failure();

  // Parse operation attributes.
  NamedAttrList attrs;
  if (parser.parseOptionalAttrDictWithKeyword(attrs))
    return failure();

  SmallVector<Type> argTypes;
  for (auto &arg : entryArgs)
    argTypes.push_back(arg.type);

  result.addAttributes(attrs);
  result.addAttribute(
      "type", TypeAttr::get(builder.getFunctionType(argTypes, resultTypes)));

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, /*arguments=*/entryArgs,
                         /*enableNameShadowing=*/false))
    return failure();
  return success();
}

void ClosureOp::print(OpAsmPrinter &p) {
  FunctionType type = getFunctionType();
  function_interface_impl::printFunctionSignature(p, *this, type.getInputs(),
                                                  /*isVariadic=*/false,
                                                  type.getResults());
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(), {"type"});
  p.printRegion(body(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

Region &ClosureOp::getLoopBody() { return body(); }

#define GET_OP_CLASSES
#include "Standalone/StdxOps.cpp.inc"
