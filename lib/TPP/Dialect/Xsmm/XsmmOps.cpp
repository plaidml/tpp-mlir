//===- XsmmOps.cpp - Xsmm dialect ops ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Xsmm/XsmmOps.h"
#include "TPP/Dialect/Xsmm/XsmmEnum.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "TPP/Dialect/Xsmm/XsmmOps.cpp.inc"

using namespace mlir;
using namespace mlir::xsmm;

template <typename EnumClass>
static ParseResult parseEnum(EnumClass &value, OpAsmParser &parser) {
  StringRef flag;
  auto loc = parser.getCurrentLocation();
  if (parser.parseKeyword(&flag))
    return failure();
  auto flagAttr = symbolizeEnum<EnumClass>(flag);
  if (!flagAttr)
    return parser.emitError(loc, "invalid enum") << flag;
  value = *flagAttr;
  return success();
}

static ParseResult parserImpl(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  // Parse the input
  result.addAttribute("inputs", DenseI64ArrayAttr::parse(parser, Type{}));

  if (parser.parseKeyword("flags") || parser.parseEqual() ||
      parser.parseLParen())
    return failure();

  // Parse flags
  SmallVector<Attribute, 4> flags;
  auto parseFlags = [&]() -> ParseResult {
    GemmFlags flag;
    if (parseEnum(flag, parser))
      return failure();
    flags.push_back(builder.getI64IntegerAttr(static_cast<int64_t>(flag)));
    return success();
  };
  if (parser.parseCommaSeparatedList(parseFlags) || parser.parseRParen())
    return failure();
  result.addAttribute("flags", builder.getArrayAttr(flags));

  // Parse dataType
  if (parser.parseKeyword("data_type") || parser.parseEqual())
    return failure();
  DataType dataType;
  if (parseEnum(dataType, parser))
    return failure();
  result.addAttribute(
      "dataType", builder.getI64IntegerAttr(static_cast<int64_t>(dataType)));
  result.addTypes(builder.getIntegerType(64));
  return success();
}

ParseResult MatmulDispatchOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  return parserImpl(parser, result);
}

ParseResult BrgemmDispatchOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  return parserImpl(parser, result);
}

template <typename OpTy>
static void printerImpl(OpAsmPrinter &printer, OpTy op) {
  printer << " [" << op.getInputs() << ']';
  printer << " "
          << " flags = (";
  llvm::interleaveComma(op.getFlags(), printer, [&](auto &attr) {
    auto flag = *symbolizeGemmFlags(attr.template cast<IntegerAttr>().getInt());
    printer << xsmm::stringifyGemmFlags(flag);
  });
  printer << ") data_type = ";
  auto dataType = op.getDataType();
  printer << xsmm::stringifyDataType(dataType);
}

void MatmulDispatchOp::print(OpAsmPrinter &printer) {
  printerImpl<MatmulDispatchOp>(printer, *this);
}

void BrgemmDispatchOp::print(OpAsmPrinter &printer) {
  printerImpl<BrgemmDispatchOp>(printer, *this);
}
