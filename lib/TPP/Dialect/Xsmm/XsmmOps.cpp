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
    return parser.emitError(loc, "invalid enum ") << flag;
  value = *flagAttr;
  return success();
}

static ParseResult parserImpl(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  // Parse the input
  DenseI64ArrayAttr kindAttr;
  if (parser.parseCustomAttributeWithFallback(kindAttr, Type{}, "inputs",
                                              result.attributes)) {
    return failure();
  }

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
  result.addAttribute("dataType",
                      DataTypeAttr::get(builder.getContext(), dataType));
  result.addTypes(builder.getIntegerType(64));

  // Parse the optional attribute list
  return parser.parseOptionalAttrDict(result.attributes);
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
  printer.printOptionalAttrDict(
      op->getAttrs(), /*elidedAttrs=*/{"dataType", "flags", "inputs"});
}

void MatmulDispatchOp::print(OpAsmPrinter &printer) {
  printerImpl<MatmulDispatchOp>(printer, *this);
}

void BrgemmDispatchOp::print(OpAsmPrinter &printer) {
  printerImpl<BrgemmDispatchOp>(printer, *this);
}

static LogicalResult verifyGemmFlags(ArrayAttr flags, DataType dataType,
                                     Operation *op) {
  SmallVector<int64_t> flagsAsInt;
  for (auto flag : flags) {
    flagsAsInt.push_back(flag.cast<IntegerAttr>().getInt());
  }
  // check uniqueness
  std::sort(flagsAsInt.begin(), flagsAsInt.end());
  auto *it = std::unique(flagsAsInt.begin(), flagsAsInt.end());
  if (it != flagsAsInt.end())
    return op->emitOpError() << "expected flags to be unique";
  // none flag conflicts with all the others
  if (llvm::is_contained(flagsAsInt, static_cast<int64_t>(GemmFlags::NONE)) &&
      flagsAsInt.size() != 1) {
    return op->emitOpError() << "'none' flags conflicts with others";
  }
  // VNNI flags must be specified only for bf16 type
  if (dataType != DataType::BF16 && llvm::any_of(flagsAsInt, [](int64_t flag) {
        return (flag == static_cast<int64_t>(GemmFlags::VNNI_B) ||
                flag == static_cast<int64_t>(GemmFlags::VNNI_A) ||
                flag == static_cast<int64_t>(GemmFlags::VNNI_C));
      })) {
    return op->emitOpError() << "VNNI flags but type is not bf16";
  }
  return success();
}

template <typename OpTy>
static LogicalResult verifyInputs(OpTy op, size_t expected) {
  // `inputs` are leading dimensions and sizes
  size_t numInputs = op.getInputs().size();
  if (numInputs != expected) {
    return op.emitOpError()
           << "expect " << expected << " args but got: " << numInputs;
  }
  return success();
}

template <typename OpTy> static LogicalResult verifyGemmLikeOp(OpTy op) {
  // 'inputs' = [m, n, k, lda, ldb, ldc]
  if (failed(verifyInputs(op, /*expected=*/6)))
    return failure();
  return verifyGemmFlags(op.getFlags(), op.getDataType(), op);
}

LogicalResult MatmulDispatchOp::verify() {
  return verifyGemmLikeOp<MatmulDispatchOp>(*this);
}

LogicalResult BrgemmDispatchOp::verify() {
  return verifyGemmLikeOp<BrgemmDispatchOp>(*this);
}

LogicalResult UnaryDispatchOp::verify() {
  // 'inputs' = [m, n, lda, ldo]
  if (failed(verifyInputs(*this, /*expected=*/4)))
    return failure();
  return success();
}

LogicalResult BinaryDispatchOp::verify() {
  // 'inputs' = [m, n, lda, ldb, ldo]
  if (failed(verifyInputs(*this, /*expected=*/5)))
    return failure();
  return success();
}

LogicalResult TernaryDispatchOp::verify() {
  // 'inputs' = [m, n, k, lda, ldb, ldc]
  if (failed(verifyInputs(*this, /*expected=*/6)))
    return failure();
  return success();
}
