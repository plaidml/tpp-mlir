//===- XeGPUOps.cpp - XeGPU dialect -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the XeGPU dialect and its basic operations.
///
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/XeGPU/IR/XeGPUOps.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include <numeric>
#include <type_traits>

#include "TPP/Dialect/XeGPU/XeUtils.h"

namespace mlir {
namespace xegpu {

const int MAX_2D_BLOCK_WIDTH_IN_ELEMENTS = 64;
const int MIN_2D_BLOCK_WIDTH_IN_ELEMENTS = 1;
const int MAX_2D_BLOCK_HEIGHT_IN_ELEMENTS = 32;
const int MIN_2D_BLOCK_HEIGHT_IN_ELEMENTS = 1;
// TODO: Generalize shapes for different architecture.
const int MAX_TM_SIZE = 8;
const int TN_SIZE = 16;
const int TK_SIZE_FOR_D16 = 16;
const int TK_SIZE_FOR_D8 = 32;

static bool vnniVerifier(size_t axis, llvm::ArrayRef<int64_t> tdescShape,
                         llvm::ArrayRef<int64_t> valueShape,
                         size_t elemTyBitWidth) {
  bool isValid = valueShape.size() == tdescShape.size() + 1;

  for (size_t i = 0; i < tdescShape.size(); i++) {
    if ((i == axis && valueShape[i] * valueShape.back() != tdescShape[i]) ||
        (i != axis && valueShape[i] != tdescShape[i]))
      isValid = false;
  }

  const static size_t xeSIMDLaneBitWidth = 32;
  auto vnni_factor = valueShape.back();

  isValid &= vnni_factor == xeSIMDLaneBitWidth / elemTyBitWidth;

  return isValid;
}

bool dpasSupportedTypes(mlir::Type type, bool isResult) {
  if (isResult) {
    if (type.isF32() || type.isInteger(32))
      return true;
    else
      return false;
  } else {
    if (type.isF16() || type.isBF16() || type.isInteger(8))
      return true;
    else
      return false;
  }
}

bool dpasSupportedShapes(DpasOp op) {

  mlir::Type lhsElemType = op.getLhsType().getElementType();
  // TODO: handle dynamic shape cast
  auto lhsShape = op.getLhsType().cast<mlir::ShapedType>().getShape();
  auto rhsShape = op.getRhsType().cast<mlir::ShapedType>().getShape();
  // Retrieve 2D shapes(MxK * KxN) from 3D. Verify this
  auto m = lhsShape[0];
  auto k = lhsShape[1] * lhsShape[2];
  auto n = rhsShape[1];

  if ((lhsElemType.isF16() || lhsElemType.isBF16()) && m <= MAX_TM_SIZE &&
      n == TN_SIZE && k == TK_SIZE_FOR_D16) {
    return true;
  }

  if (lhsElemType.isInteger(8) && m <= MAX_TM_SIZE && n == TN_SIZE &&
      k == TK_SIZE_FOR_D8) {
    return true;
  }

  return false;
}

template <typename CustomEnum, typename CustomEnumAttr>
static mlir::ParseResult parseCustomEnumAttr(mlir::OpAsmParser &parser,
                                             mlir::OperationState &result,
                                             llvm::StringRef attrKeyword) {
  auto loc = parser.getCurrentLocation();
  auto attrOptional = mlir::FieldParser<CustomEnum, CustomEnum>::parse(parser);
  if (mlir::failed(attrOptional))
    return parser.emitError(loc, "invalid ")
           << "memory_scope attribute specification";
  auto attr =
      CustomEnumAttr::get(parser.getBuilder().getContext(), *attrOptional);
  result.addAttribute(attrKeyword, attr);
  return mlir::success();
}

template <typename AttrType>
static mlir::ParseResult parseBoolAndIntegerAttr(mlir::OpAsmParser &parser,
                                                 mlir::OperationState &result,
                                                 llvm::StringRef attrKeyword) {
  AttrType attr;
  mlir::Type ty;

  if (std::is_same<AttrType, mlir::BoolAttr>::value) {
    ty = parser.getBuilder().getIntegerType(1);

  } else if (std::is_same<AttrType, mlir::IntegerAttr>::value) {
    ty = parser.getBuilder().getIntegerType(32);
  } else if (std::is_same<AttrType, mlir::DenseI64ArrayAttr>::value) {
    ty = mlir::Type{};
  } else {
    assert(0 && "Unreachable.\n");
  }

  if (parser.parseCustomAttributeWithFallback(attr, ty))
    return mlir::failure();

  if (attr)
    result.addAttribute(attrKeyword, attr);
  return mlir::success();
};

/// @brief Parsing optional attribute list which are enclosed in braces "{}",
/// and seperated by comma
/// @param parser
/// @param result
/// @param allowedKeywords
/// @return
static mlir::ParseResult
parseOptionalAttrDict(mlir::OpAsmParser &parser, mlir::OperationState &result,
                      llvm::ArrayRef<llvm::StringRef> allowedKeywords,
                      bool isWrite = false) {
  // no optional attributes, return success
  if (mlir::failed(parser.parseOptionalLBrace()))
    return mlir::success();

  auto parseElt = [&]() -> mlir::ParseResult {
    auto loc = parser.getCurrentLocation();
    llvm::StringRef nameId;
    if (parser.parseOptionalKeyword(&nameId, allowedKeywords))
      return parser.emitError(loc, "invalid")
             << "attribute keyword: " << nameId << ".\n";

    if (parser.parseEqual())
      return ::mlir::failure();

    if (nameId == "memory_scope")
      return parseCustomEnumAttr<MemoryScope, MemoryScopeAttr>(parser, result,
                                                               nameId);

    if (nameId == "l1_hint" || nameId == "l2_hint" || nameId == "l3_hint") {
      if (isWrite)
        return parseCustomEnumAttr<CacheWriteHint, CacheWriteHintAttr>(
            parser, result, nameId);
      else
        return parseCustomEnumAttr<CacheReadHint, CacheReadHintAttr>(
            parser, result, nameId);
    }

    if (nameId == "chunk_size_per_lane" || nameId == "vnni_axis")
      return parseBoolAndIntegerAttr<mlir::IntegerAttr>(parser, result, nameId);

    if (nameId == "boundary_check")
      return parseBoolAndIntegerAttr<mlir::BoolAttr>(parser, result, nameId);

    if (nameId == "transpose")
      return parseBoolAndIntegerAttr<mlir::DenseI64ArrayAttr>(parser, result,
                                                              nameId);

    assert(0 && "Unreachable!");
  };

  if (parser.parseCommaSeparatedList(parseElt))
    return mlir::failure();

  return parser.parseRBrace();
}

template <typename T>
static void printCacheHintAttrs(mlir::OpAsmPrinter &printer, T op,
                                bool printSep) {
  if (op.getL1HintAttr()) {
    if (printSep)
      printer << ", ";
    printer << "l1_hint = " << op.getL1Hint().value();
    printSep = true;
  }

  if (op.getL2HintAttr()) {
    if (printSep)
      printer << ", ";
    printer << "l2_hint = " << op.getL2Hint().value();
    printSep = true;
  }

  if (op.getL3HintAttr()) {
    if (printSep)
      printer << ", ";
    printer << "l3_hint = " << op.getL3Hint().value();
  }
}

mlir::ParseResult CreateNdDescOp::parse(mlir::OpAsmParser &parser,
                                        mlir::OperationState &result) {

  // parse the source operand
  mlir::OpAsmParser::UnresolvedOperand sourceRawOperands[1];
  llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> sourceOperands(
      sourceRawOperands);
  llvm::SMLoc sourceOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceRawOperands[0]))
    return ::mlir::failure();

  // parse the offset operand, in format of [x, y]
  llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> offsetsOperands;
  mlir::DenseI64ArrayAttr static_offsetsAttr;
  llvm::SMLoc offsetsOperandsLoc = parser.getCurrentLocation();
  if (parseDynamicIndexList(parser, offsetsOperands, static_offsetsAttr))
    return ::mlir::failure();
  result.addAttribute("static_offsets", static_offsetsAttr);

  llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> shapeOperands;
  llvm::SMLoc shapeOperandsLoc;

  llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> stridesOperands;
  llvm::SMLoc stridesOperandsLoc;
  // parse optional shape and strides, shape and strides should always come
  // together
  if (::mlir::succeeded(parser.parseOptionalComma())) {
    // parse shape part, in form of [x, y]
    if (parser.parseLSquare())
      return mlir::failure();
    shapeOperandsLoc = parser.getCurrentLocation();
    if (parser.parseOperandList(shapeOperands))
      return mlir::failure();
    if (parser.parseRSquare())
      return mlir::failure();

    if (parser.parseComma())
      return mlir::failure();

    // parse stride part, in form of [x, y]
    if (parser.parseLSquare())
      return ::mlir::failure();
    stridesOperandsLoc = parser.getCurrentLocation();
    if (parser.parseOperandList(stridesOperands))
      return mlir::failure();
    if (parser.parseRSquare())
      return ::mlir::failure();
  }

  if (parseOptionalAttrDict(parser, result, {"memory_scope", "boundary_check"}))
    return mlir::failure();

  if (parser.parseColon())
    return ::mlir::failure();

  mlir::Type sourceRawTypes[1];
  llvm::ArrayRef<::mlir::Type> sourceTypes(sourceRawTypes);
  if (parser.parseType(sourceRawTypes[0]))
    return ::mlir::failure();

  if (parser.parseArrow())
    return ::mlir::failure();

  mlir::Type TensorDescRawTypes[1];
  llvm::ArrayRef<::mlir::Type> TensorDescTypes(TensorDescRawTypes);
  if (parser.parseType(TensorDescRawTypes[0]))
    return ::mlir::failure();
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {1, static_cast<int32_t>(offsetsOperands.size()),
                           static_cast<int32_t>(shapeOperands.size()),
                           static_cast<int32_t>(stridesOperands.size())}));

  mlir::Type indexType = parser.getBuilder().getIndexType();
  result.addTypes(TensorDescTypes);
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc,
                             result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(offsetsOperands, indexType, offsetsOperandsLoc,
                             result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(shapeOperands, indexType, shapeOperandsLoc,
                             result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(stridesOperands, indexType, stridesOperandsLoc,
                             result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CreateNdDescOp::print(::mlir::OpAsmPrinter &printer) {
  printer << ' ';
  printer << getSource();
  printDynamicIndexList(printer, *this, getOffsets(), getStaticOffsetsAttr());
  if (!getShape().empty()) {
    printer << ",";
    printer << ' ' << "[";
    printer << getShape();
    printer << "]";
  }

  if (!getStrides().empty()) {
    printer << ",";
    printer << ' ' << "[";
    printer << getStrides();
    printer << "]";
  }

  // printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  printer << ' ' << "{";
  printer << "memory_scope = " << getMemoryScope();
  printer << "," << ' ';
  printer << "boundary_check = " << getBoundaryCheck();
  printer << "}";

  printer << ' ' << ":";
  printer << ' ';
  printer << getSource().getType();
  printer << ' ' << "->";
  printer << ' ';
  printer << getTensorDesc().getType();
}

static std::string getValueAsString(::mlir::Value op, bool asOperand = false) {
  std::string buf;
  buf.clear();
  llvm::raw_string_ostream os(buf);
  auto flags = ::mlir::OpPrintingFlags().assumeVerified();
  if (asOperand)
    op.printAsOperand(os, flags);
  else
    op.print(os, flags);
  os.flush();
  return buf;
}

mlir::LogicalResult CreateNdDescOp::verify() {
  llvm::dbgs() << "Op: " << getValueAsString(*this)
               << "\n\tstatic offsets: " << makeString(getStaticOffsets())
               << "\n\n";
  // it is invalid to have both dynamic and static shape
  if (!(hasDynamicShape() ^ hasStaticShape()))
    return emitOpError("It is invalid to have both or none of dynamic shape "
                       "and static shape. Only one of them is needed.");

  if (getOffsetsRank() != getShapeRank() ||
      getShapeRank() != getStridesRank() ||
      (isMemRef(getSource()) && getRankOf(getSource()) != getOffsetsRank()))
    return emitOpError("Expecting the rank of shape, strides and offsets "
                       "should match with each other.");

  return mlir::success();
}

mlir::ParseResult CreateDescOp::parse(mlir::OpAsmParser &parser,
                                      mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand sourceRawOperands[1];
  llvm::ArrayRef<mlir::OpAsmParser::UnresolvedOperand> sourceOperands(
      sourceRawOperands);
  llvm::SMLoc sourceOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceRawOperands[0]))
    return mlir::failure();

  if (parser.parseComma())
    return mlir::failure();

  mlir::OpAsmParser::UnresolvedOperand offsetsRawOperands[1];
  llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> offsetsOperands(
      offsetsRawOperands);
  llvm::SMLoc offsetsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(offsetsRawOperands[0]))
    return mlir::failure();

  if (parseOptionalAttrDict(parser, result,
                            {"memory_scope", "chunk_size_per_lane"}))
    return mlir::failure();

  if (parser.parseColon())
    return ::mlir::failure();

  mlir::Type sourceRawTypes[1];
  llvm::ArrayRef<mlir::Type> sourceTypes(sourceRawTypes);
  if (parser.parseType(sourceRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  mlir::Type offsetsRawTypes[1];
  llvm::ArrayRef<mlir::Type> offsetsTypes(offsetsRawTypes);
  if (parser.parseType(offsetsRawTypes[0]))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  mlir::Type TensorDescRawTypes[1];
  llvm::ArrayRef<mlir::Type> TensorDescTypes(TensorDescRawTypes);
  if (parser.parseType(TensorDescRawTypes[0]))
    return ::mlir::failure();

  result.addTypes(TensorDescTypes);
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc,
                             result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(offsetsOperands, offsetsTypes, offsetsOperandsLoc,
                             result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CreateDescOp::print(::mlir::OpAsmPrinter &printer) {
  printer << ' ';
  printer << getSource();
  printer << ",";
  printer << ' ';
  printer << getOffsets();

  printer << ' ' << "{";
  printer << "memory_scope = " << getMemoryScope();
  printer << "," << ' ';
  printer << "chunk_size_per_lane = " << getChunkSizePerLane();
  printer << "}";

  printer << ' ' << ":";
  printer << ' ';
  printer << getSource().getType();
  printer << ",";
  printer << ' ';
  printer << getOffsets().getType();
  printer << ' ' << "->";
  printer << ' ';
  printer << getTensorDesc().getType();
}

mlir::LogicalResult CreateDescOp::verify() {
  auto offsetType = getOffsets().getType();
  auto tdescType = getTensorDesc().getType();
  auto chunkSize = getChunkSizePerLane();

  auto offsetShape = offsetType.getShape();
  auto tdescShape = tdescType.getShape();

  if (getRankOf(getSource()) > 1)
    return emitOpError(
        "Expecting the source is a 1D memref or pointer (uint64_t).");

  if (offsetShape.size() != 1)
    return emitOpError("Expecting the offset is a 1D vector.");

  if (offsetShape != tdescShape && (offsetShape != tdescShape.drop_back() ||
                                    tdescShape.back() != chunkSize)) {
    return emitOpError("Expecting dimensions of offsets is the same as the "
                       "tensor descriptor, or one less than.");
  }

  if (!tdescType.getEncoding())
    return emitOpError("Expecting the presence of scattered attribute for "
                       "scattered tensor descriptor.");
  return mlir::success();
}

mlir::ParseResult LoadNDOp::parse(::mlir::OpAsmParser &parser,
                                  ::mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand TensorDescRawOperands[1];
  llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> TensorDescOperands(
      TensorDescRawOperands);
  llvm::SMLoc TensorDescOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(TensorDescRawOperands[0]))
    return ::mlir::failure();

  if (parseOptionalAttrDict(
          parser, result,
          {"vnni_axis", "transpose", "l1_hint", "l2_hint", "l3_hint"}))
    return mlir::failure();

  if (parser.parseColon())
    return ::mlir::failure();

  mlir::Type TensorDescRawTypes[1];
  llvm::ArrayRef<::mlir::Type> TensorDescTypes(TensorDescRawTypes);
  if (parser.parseType(TensorDescRawTypes[0]))
    return ::mlir::failure();

  if (parser.parseArrow())
    return ::mlir::failure();

  mlir::Type valueRawTypes[1];
  llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);
  if (parser.parseType(valueRawTypes[0]))
    return ::mlir::failure();

  result.addTypes(valueTypes);
  if (parser.resolveOperands(TensorDescOperands, TensorDescTypes,
                             TensorDescOperandsLoc, result.operands))
    return mlir::failure();

  return mlir::success();
}

void LoadNDOp::print(::mlir::OpAsmPrinter &printer) {
  printer << ' ';
  printer << getTensorDesc();

  if ((*this)->getAttrs().size()) {
    bool printSep = false;
    printer << ' ' << "{";
    if (getVnniAxisAttr()) {
      printer << "vnni_axis = " << getVnniAxis().value();
      printSep = true;
    }

    if (getTransposeAttr()) {
      if (printSep)
        printer << ", ";
      printer << "transpose = ";
      getTransposeAttr().print(printer);
      printSep = true;
    }

    printCacheHintAttrs<LoadNDOp>(printer, *this, printSep);

    printer << "}";
  }
  printer << ' ' << ":";
  printer << ' ';
  printer << getTensorDesc().getType();
  printer << ' ' << "->";
  printer << ' ';
  printer << getValue().getType();
}

mlir::LogicalResult LoadNDOp::verify() {
  auto input = getTensorDesc();

  auto tdescShape = getTensorDesc().getType().getShape().vec();
  auto valueShape = getValue().getType().getShape().vec();

  auto tdescElemTy = getTensorDesc().getType().getElementType();
  auto valueElemTy = getValue().getType().getElementType();

  if (!llvm::isa<TensorDescType>(input.getType()) ||
      input.getType().getRank() != 2) {
    return emitOpError("The input to LoadNDOp should be a 2D TensorDesc.");
  }

  if (tdescElemTy != valueElemTy)
    return emitOpError(
        "Value should have the same element type as TensorDesc.");

  if (!tdescElemTy.isIntOrFloat()) {
    // FIXME: currently only int and float type are supported for estimating
    // size info improve it to make it more robust if neccessary
    return emitOpError(
        "Currently only IntType or FloatType are supported for Load2DOp.");
  }

  auto width = input.getType().getShape()[1];
  auto height = input.getType().getShape()[0];
  auto elemTyByteWidth = tdescElemTy.getIntOrFloatBitWidth() / 8;

  if (width < MIN_2D_BLOCK_WIDTH_IN_ELEMENTS ||
      width > MAX_2D_BLOCK_WIDTH_IN_ELEMENTS ||
      (width * elemTyByteWidth) % 4 != 0) {
    return emitOpError("Invalid width size for 2D block load.  \
                        The specification expects the value to \
                        be in range [1, 64], and The the total \
                        data size (width * elemTyBytes) to be multiple of 4.\n");
  }

  if (height < MIN_2D_BLOCK_HEIGHT_IN_ELEMENTS ||
      height > MAX_2D_BLOCK_HEIGHT_IN_ELEMENTS) {
    return emitOpError(
        "Invalid height size for 2D block load. The specification expects the "
        "value to be in range [1, 32].\n");
  }

  if (getTranspose()) {
    auto dim0 = getTranspose().value()[0];
    auto dim1 = getTranspose().value()[1];
    auto tmp = valueShape[dim0];
    valueShape[dim0] = valueShape[dim1];
    valueShape[dim1] = tmp;
  }

  if (!getVnniAxis()) {
    if (valueShape != tdescShape)
      return emitOpError("Value should have the same shape as TensorDesc when "
                         "vnni is not enabled.");
  } else {
    auto axis = getVnniAxis().value();
    auto bits = getTensorDesc().getType().getElementTypeBitWidth();
    if (!vnniVerifier(axis, tdescShape, valueShape, bits))
      return emitOpError("Invalid vnni transform. When vnni is enabled, value "
                         "should have one more"
                         "dimention than the TensorDesc, but having same "
                         "number of data elements."
                         "Also, vnni factor should be calculated as "
                         "simd_lane_width / elementTypeBitWidth."
                         "For element type having more than 32 bits, vnni "
                         "shouldn't be used.\n");
  }
  return mlir::success();
}

::mlir::ParseResult StoreNDOp::parse(::mlir::OpAsmParser &parser,
                                     ::mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(
      valueRawOperands);
  llvm::SMLoc valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return mlir::failure();

  if (parser.parseComma())
    return mlir::failure();

  mlir::OpAsmParser::UnresolvedOperand TensorDescRawOperands[1];
  llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> TensorDescOperands(
      TensorDescRawOperands);
  llvm::SMLoc TensorDescOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(TensorDescRawOperands[0]))
    return mlir::failure();

  if (parseOptionalAttrDict(parser, result, {"l1_hint", "l2_hint", "l3_hint"},
                            true))
    return mlir::failure();

  if (parser.parseColon())
    return ::mlir::failure();

  mlir::Type valueRawTypes[1];
  llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);
  if (parser.parseType(valueRawTypes[0]))
    return ::mlir::failure();

  if (parser.parseComma())
    return ::mlir::failure();

  mlir::Type TensorDescRawTypes[1];
  llvm::ArrayRef<::mlir::Type> TensorDescTypes(TensorDescRawTypes);
  if (parser.parseType(TensorDescRawTypes[0]))
    return ::mlir::failure();

  if (parser.resolveOperands(TensorDescOperands, TensorDescTypes,
                             TensorDescOperandsLoc, result.operands))
    return ::mlir::failure();

  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc,
                             result.operands))
    return ::mlir::failure();

  return ::mlir::success();
}

void StoreNDOp::print(::mlir::OpAsmPrinter &printer) {
  printer << ' ';
  printer << getValue();
  printer << ",";
  printer << ' ';
  printer << getTensorDesc();
  if ((*this)->getAttrs().size()) {
    bool printSep = false;
    printer << ' ' << "{";
    printCacheHintAttrs<StoreNDOp>(printer, *this, printSep);
    printer << "}";
  }
  printer << ' ' << ":";
  printer << ' ';
  printer << getValue().getType();
  printer << ",";
  printer << ' ';
  printer << getTensorDesc().getType();
}

mlir::LogicalResult StoreNDOp::verify() {
  auto dst = getTensorDesc(); // Tile
  auto val = getValue();      // Vector

  if (dst.getType().getShape() != val.getType().getShape()) {
    return emitOpError(
        "The value (vector) shape doesn't match the memory (dst) shape.\n");
  }

  auto dstElemTy = dst.getType().getElementType();
  auto valElemTy = val.getType().getElementType();

  if (dstElemTy != valElemTy) {
    return emitOpError("The elem type of value (vector) shape doesn't match "
                       "the elem type of memory (dst) shape.\n");
  }

  if (!dstElemTy.isIntOrFloat()) {
    // FIXME: currently only int and float type are supported for estimating
    // size info improve it to make it more robust if neccessary
    return emitOpError(
        "Currently only IntType or FloatType are supported for Store2DOp.");
  }

  auto width = dst.getType().getShape()[1];
  auto height = dst.getType().getShape()[0];
  auto elemTyByteWidth = dstElemTy.getIntOrFloatBitWidth() / 8;

  if (width < MIN_2D_BLOCK_WIDTH_IN_ELEMENTS ||
      width > MAX_2D_BLOCK_WIDTH_IN_ELEMENTS ||
      (width * elemTyByteWidth) % 4 != 0) {
    return emitOpError("Invalid width size for 2D block write. \
                        The specification expects the value to \
                        be in range [1, 64], and The the total \
                        data size (width * elemTyBytes) to be multiple of 4.\n");
  }

  if (height < MIN_2D_BLOCK_HEIGHT_IN_ELEMENTS ||
      height > MAX_2D_BLOCK_HEIGHT_IN_ELEMENTS) {
    return emitOpError(
        "Invalid height size for 2D block write. The specification expects the "
        "value to be in range [1, 32].\n");
  }

  return mlir::success();
}

::mlir::ParseResult PrefetchNDOp::parse(::mlir::OpAsmParser &parser,
                                        ::mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand TensorDescRawOperands[1];
  llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> TensorDescOperands(
      TensorDescRawOperands);
  llvm::SMLoc TensorDescOperandsLoc;
  mlir::Type TensorDescRawTypes[1];
  llvm::ArrayRef<::mlir::Type> TensorDescTypes(TensorDescRawTypes);

  TensorDescOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(TensorDescRawOperands[0]))
    return ::mlir::failure();

  if (parseOptionalAttrDict(parser, result, {"l1_hint", "l2_hint", "l3_hint"}))
    return mlir::failure();

  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(TensorDescRawTypes[0]))
    return ::mlir::failure();
  if (parser.resolveOperands(TensorDescOperands, TensorDescTypes,
                             TensorDescOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void PrefetchNDOp::print(::mlir::OpAsmPrinter &printer) {
  printer << ' ';
  printer << getTensorDesc();
  // printer.printOptionalAttrDict((*this)->getAttrs());
  if ((*this)->getAttrs().size()) {
    bool printSep = false;
    printer << ' ' << "{";
    printCacheHintAttrs<PrefetchNDOp>(printer, *this, printSep);
    printer << "}";
  }

  printer << ' ' << ":";
  printer << ' ';
  printer << getTensorDesc().getType();
}

mlir::LogicalResult DpasOp::verify() {

  int64_t lhsRank = getLhsType().getRank();
  int64_t rhsRank = getRhsType().getRank();
  mlir::Type lhsElemType = getLhsType().getElementType();
  mlir::Type rhsElemType = getRhsType().getElementType();
  mlir::Type resultElemType = getResultType().getElementType();

  if (!dpasSupportedTypes(lhsElemType, 0)) {
    return emitOpError("Unsupported src datatype for dpas op");
  }

  if (!dpasSupportedTypes(resultElemType, 1)) {
    return emitOpError("Unsupported result datatype for dpas op");
  }

  if (lhsElemType != rhsElemType) {
    return emitOpError("lhs and rhs element type does not match for dpas op");
  }

  if (getAcc()) {
    mlir::Type accElemType = getAccType().getElementType();
    if (accElemType != resultElemType) {
      return emitOpError(
          "Accumulator and Result element type does not match for dpas op");
    }
  }

  if (!dpasSupportedShapes(*this)) {
    return emitOpError("Incorrect shapes for dpas op");
  }

  if (lhsRank != rhsRank) {
    return emitOpError("lhs and rhs rank does not match for dpas op");
  }

  if (lhsRank < 3) {
    return emitOpError("dpas op requires 3d vector. Rank is not 3");
  }

  return mlir::success();
}

::mlir::ParseResult LoadGatherOp::parse(::mlir::OpAsmParser &parser,
                                        ::mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand TensorDescRawOperands[1];
  llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> TensorDescOperands(
      TensorDescRawOperands);
  llvm::SMLoc TensorDescOperandsLoc;
  mlir::OpAsmParser::UnresolvedOperand maskRawOperands[1];
  llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> maskOperands(
      maskRawOperands);
  llvm::SMLoc maskOperandsLoc;

  mlir::Type TensorDescRawTypes[1];
  llvm::ArrayRef<::mlir::Type> TensorDescTypes(TensorDescRawTypes);
  mlir::Type maskRawTypes[1];
  llvm::ArrayRef<mlir::Type> maskTypes(maskRawTypes);
  mlir::Type valueRawTypes[1];
  llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);

  TensorDescOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(TensorDescRawOperands[0]))
    return ::mlir::failure();

  if (parser.parseComma())
    return ::mlir::failure();

  maskOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(maskRawOperands[0]))
    return mlir::failure();

  if (parseOptionalAttrDict(
          parser, result,
          {"vnni_axis", "transpose", "l1_hint", "l2_hint", "l3_hint"}))
    return mlir::failure();

  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(TensorDescRawTypes[0]))
    return ::mlir::failure();

  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(maskRawTypes[0]))
    return ::mlir::failure();

  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseType(valueRawTypes[0]))
    return ::mlir::failure();

  result.addTypes(valueTypes);

  if (parser.resolveOperands(TensorDescOperands, TensorDescTypes,
                             TensorDescOperandsLoc, result.operands))
    return ::mlir::failure();

  if (parser.resolveOperands(maskOperands, maskTypes, maskOperandsLoc,
                             result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void LoadGatherOp::print(mlir::OpAsmPrinter &printer) {
  printer << ' ';
  printer << getTensorDesc();
  printer << ",";
  printer << ' ';
  printer << getMask();
  if ((*this)->getAttrs().size()) {
    bool printSep = false;
    printer << ' ' << "{";
    if (getVnniAxisAttr()) {
      printer << "vnni_axis = " << getVnniAxis().value();
      printSep = true;
    }

    if (getTransposeAttr()) {
      if (printSep)
        printer << ", ";
      printer << "transpose = ";
      getTransposeAttr().print(printer);
      printSep = true;
    }

    printCacheHintAttrs<LoadGatherOp>(printer, *this, printSep);

    printer << "}";
  }

  printer << ' ' << ":";
  printer << ' ';
  printer << getTensorDesc().getType();
  printer << ",";
  printer << ' ';
  printer << getMask().getType();
  printer << ' ' << "->";
  printer << ' ';
  printer << getValue().getType();
}

mlir::LogicalResult LoadGatherOp::verify() {
  // length of the offsets vector must match the dim-0 of the tensor descriptor
  auto tdescShape = getTensorDesc().getType().getShape().vec();
  auto maskShape = getMask().getType().getShape().vec();
  auto valueShape = getValue().getType().getShape().vec();

  auto tdescElemTy = getTensorDesc().getType().getElementType();
  auto valueElemTy = getValue().getType().getElementType();

  if (tdescShape != maskShape)
    return emitOpError("Mask should have the same shape as TensorDesc.");

  if (tdescElemTy != valueElemTy)
    return emitOpError(
        "Value should have the same element type as TensorDesc.");

  if (getTranspose()) {
    auto dim0 = getTranspose().value()[0];
    auto dim1 = getTranspose().value()[1];
    auto tmp = valueShape[dim0];
    valueShape[dim0] = valueShape[dim1];
    valueShape[dim1] = tmp;
  }

  if (!getVnniAxis()) {
    if (valueShape != tdescShape)
      return emitOpError("Value should have the same shape as TensorDesc when "
                         "vnni is not enabled.");
  } else {
    auto axis = getVnniAxis().value();
    auto bits = getTensorDesc().getType().getElementTypeBitWidth();
    if (!vnniVerifier(axis, tdescShape, valueShape, bits))
      return emitOpError("Invalid vnni transform. When vnni is enabled, value "
                         "should have one more"
                         "dimention than the TensorDesc, but having same "
                         "number of data elements."
                         "Also, vnni factor should be calculated as "
                         "simd_lane_width / elementTypeBitWidth."
                         "For element type having more than 32 bits, vnni "
                         "shouldn't be used.\n");
  }

  return ::mlir::success();
}

::mlir::ParseResult StoreScatterOp::parse(::mlir::OpAsmParser &parser,
                                          ::mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand TensorDescRawOperands[1];
  llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> TensorDescOperands(
      TensorDescRawOperands);
  llvm::SMLoc TensorDescOperandsLoc;

  mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(
      valueRawOperands);
  llvm::SMLoc valueOperandsLoc;

  mlir::OpAsmParser::UnresolvedOperand maskRawOperands[1];
  llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> maskOperands(
      maskRawOperands);
  llvm::SMLoc maskOperandsLoc;

  mlir::Type valueRawTypes[1];
  llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);

  mlir::Type TensorDescRawTypes[1];
  llvm::ArrayRef<::mlir::Type> TensorDescTypes(TensorDescRawTypes);

  mlir::Type maskRawTypes[1];
  llvm::ArrayRef<mlir::Type> maskTypes(maskRawTypes);

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();

  if (parser.parseComma())
    return ::mlir::failure();

  TensorDescOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(TensorDescRawOperands[0]))
    return ::mlir::failure();

  if (parser.parseComma())
    return ::mlir::failure();

  maskOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(maskRawOperands[0]))
    return mlir::failure();

  if (parseOptionalAttrDict(parser, result, {"l1_hint", "l2_hint", "l3_hint"},
                            true))
    return mlir::failure();

  if (parser.parseColon())
    return ::mlir::failure();

  // if (parser.parseLParen())
  //   return ::mlir::failure();

  if (parser.parseType(valueRawTypes[0]))
    return ::mlir::failure();

  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(TensorDescRawTypes[0]))
    return ::mlir::failure();

  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseType(maskRawTypes[0]))
    return ::mlir::failure();

  // if (parser.parseRParen())
  //   return ::mlir::failure();

  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc,
                             result.operands))
    return ::mlir::failure();

  if (parser.resolveOperands(TensorDescOperands, TensorDescTypes,
                             TensorDescOperandsLoc, result.operands))
    return ::mlir::failure();

  if (parser.resolveOperands(maskOperands, maskTypes, maskOperandsLoc,
                             result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void StoreScatterOp::print(::mlir::OpAsmPrinter &printer) {
  printer << ' ';
  printer << getValue();
  printer << ",";
  printer << ' ';
  printer << getTensorDesc();
  printer << ",";
  printer << ' ';
  printer << getMask();
  if ((*this)->getAttrs().size()) {
    bool printSep = false;
    printer << ' ' << "{";
    printCacheHintAttrs<StoreScatterOp>(printer, *this, printSep);
    printer << "}";
  }

  printer << ' ' << ":";
  printer << ' ';
  printer << getValue().getType();
  printer << ",";
  printer << ' ';
  printer << getTensorDesc().getType();
  printer << ",";
  printer << ' ';
  printer << getMask().getType();
}

::mlir::LogicalResult StoreScatterOp::verify() {
  // length of the offsets vector must match the dim-0 of the tensor descriptor
  if (getTensorDesc().getType().getShape() != getMask().getType().getShape()) {
    return emitOpError("Mask should have the same shape as TensorDesc.");
  }
  return ::mlir::success();
}

::mlir::LogicalResult UpdateOffsetOp::verify() {
  // length of the offsets vector must match the dim-0 of the tensor descriptor
  if (getTensorDesc().getType().getShape()[0] !=
      getOffsets().getType().getShape()[0]) {
    return emitOpError("invalid number of offsets.");
  }
  return ::mlir::success();
}

::mlir::LogicalResult UpdateNDOffsetOp::verify() {
  // number of offsets specified must match the rank of the tensor descriptor
  if (getTensorDesc().getType().getRank() !=
      static_cast<int64_t>(getOffsets().size())) {
    return emitOpError("invalid number of offsets.");
  }
  return ::mlir::success();
}

// =========================== begin for dewei ============================== //
// the following code block is pending removal by Dewei. It is from old
// definition, and is used to temporary avoid XeGPUToSPIRV compiling error.
bool BaseTensorDescType::hasRank() const { return true; }

llvm::ArrayRef<int64_t> BaseTensorDescType::getShape() const {
  return cast<TileType>().getShape();
}

BaseTensorDescType
BaseTensorDescType::cloneWith(std::optional<llvm::ArrayRef<int64_t>> shape,
                              Type elementType) const {
  return TileType::get(shape.value_or(getShape()), elementType);
}

bool BaseTensorDescType::isValidElementType(Type type) {
  return type.isIntOrIndexOrFloat() || type.isa<mlir::ComplexType>();
}
// =========================== end for dewei ============================== //

} // namespace xegpu
} // namespace mlir

#include "TPP/Dialect/XeGPU/IR/XeGPUOpsEnums.cpp.inc"
#define GET_OP_CLASSES
#include "TPP/Dialect/XeGPU/IR/XeGPUOps.cpp.inc"
