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
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/TypeUtilities.h"

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

void XeGPUDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include <TPP/Dialect/XeGPU/IR/XeGPUOpsTypes.cpp.inc>
      >();
  addOperations<
#define GET_OP_LIST
#include <TPP/Dialect/XeGPU/IR/XeGPUOps.cpp.inc>
      >();
}

bool TileBase::hasRank() const { return true; }

llvm::ArrayRef<int64_t> TileBase::getShape() const {
  return cast<TileType>().getShape();
}

TileBase TileBase::cloneWith(std::optional<llvm::ArrayRef<int64_t>> shape,
                             Type elementType) const {
  return TileType::get(shape.value_or(getShape()), elementType);
}

bool TileBase::isValidElementType(Type type) {
  return type.isIntOrIndexOrFloat() || type.isa<mlir::ComplexType>();
}

static mlir::LogicalResult parseShape(mlir::AsmParser &parser,
                                      llvm::SmallVector<int64_t> &shape,
                                      mlir::Type &type) {
  llvm::SmallVector<int64_t> dimensions;
  if (parser.parseDimensionList(dimensions))
    return mlir::failure();

  mlir::Type t;
  if (parser.parseType(t))
    return mlir::failure();

  shape = std::move(dimensions);
  type = std::move(t);
  return mlir::success();
}

static void printShape(mlir::AsmPrinter &printer, llvm::ArrayRef<int64_t> shape,
                       mlir::Type type) {
  for (int64_t dim : shape) {
    if (mlir::ShapedType::isDynamic(dim))
      printer << '?';
    else
      printer << dim;
    printer << 'x';
  }
  printer << type;
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

mlir::LogicalResult Load2DOp::verify() {
  auto input = getTile();

  if (!llvm::isa<TileType>(input.getType()) || input.getType().getRank() != 2) {
    return emitOpError("The input to Load2DOp should be a 2D tile");
  }

  auto elemTy = input.getType().getElementType();

  if (!elemTy.isIntOrFloat()) {
    // FIXME: currently only int and float type are supported for estimating
    // size info improve it to make it more robust if neccessary
    return emitOpError(
        "Currently only IntType or FloatType are supported for Load2DOp.");
  }

  auto width = input.getType().getShape()[1];
  auto height = input.getType().getShape()[0];
  auto elemTyByteWidth = elemTy.getIntOrFloatBitWidth() / 8;

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

  return mlir::success();
}

mlir::LogicalResult Store2DOp::verify() {
  auto dst = getTile();  // Tile
  auto val = getValue(); // Vector

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

} // namespace xegpu
} // namespace mlir

#include <TPP/Dialect/XeGPU/IR/XeGPUOpsDialect.cpp.inc>
#define GET_TYPEDEF_CLASSES
#include <TPP/Dialect/XeGPU/IR/XeGPUOpsTypes.cpp.inc>
#define GET_OP_CLASSES
#include <TPP/Dialect/XeGPU/IR/XeGPUOps.cpp.inc>
