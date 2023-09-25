//===- XeGPUDialect.cpp - XeGPU dialect -------*- C++ -*-===//
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

#include "TPP/Dialect/XeGPU/XeUtils.h"

namespace mlir {
namespace xegpu {

void XeGPUDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "TPP/Dialect/XeGPU/IR/XeGPUOpsTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "TPP/Dialect/XeGPU/IR/XeGPUOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "TPP/Dialect/XeGPU/IR/XeGPUOpsAttrs.cpp.inc"
      >();
}

// custom parser for XeGPU_TensorDesc (shape and type parameter)
static mlir::LogicalResult parseShapeAndType(mlir::AsmParser &parser,
                                             llvm::SmallVector<int64_t> &shape,
                                             mlir::Type &type) {
  llvm::SmallVector<int64_t> dimensions;
  if (parser.parseDimensionList(dimensions))
    return mlir::failure();
  shape = std::move(dimensions);

  mlir::Type t;
  if (parser.parseType(t))
    return mlir::failure();
  type = std::move(t);

  return mlir::success();
}

// custom printer for XeGPU_TensorDesc (shape and type parameter)
static void printShapeAndType(mlir::AsmPrinter &printer,
                              llvm::ArrayRef<int64_t> shape, mlir::Type type) {
  for (int64_t dim : shape) {
    if (mlir::ShapedType::isDynamic(dim))
      printer << '?';
    else
      printer << dim;
    printer << 'x';
  }
  printer << type;
}

} // namespace xegpu
} // namespace mlir

#include "TPP/Dialect/XeGPU/IR/XeGPUOpsDialect.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "TPP/Dialect/XeGPU/IR/XeGPUOpsAttrs.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "TPP/Dialect/XeGPU/IR/XeGPUOpsTypes.cpp.inc"
