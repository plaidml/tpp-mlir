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

#include "TPP/Dialect/XeGPU/DebugUtils.h"

namespace imex {
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

bool printDefaultValues() {
  auto *env = getenv("IMEX_XEGPU_PRINT_DEFAULTS");
  if (env && std::string(env) == "true")
    return true;
  return false;
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

// custom parser for XeGPU_TensorDesc (scope, encoding and mapping parameter)
static mlir::LogicalResult parseTensorDescAttr(mlir::AsmParser &parser,
                                               imex::xegpu::MemoryScope &scope,
                                               mlir::Attribute &encoding,
                                               mlir::Attribute &mapping) {
  // implies no attrbutes
  if (mlir::failed(parser.parseOptionalComma()))
    return mlir::success();

  auto parseElt = [&]() -> mlir::ParseResult {
    llvm::StringRef nameId;

    if (!parser.parseOptionalKeyword(&nameId, {"memory_scope"})) {
      auto loc = parser.getCurrentLocation();
      if (parser.parseEqual())
        return mlir::failure();

      auto attrOptional =
          ::mlir::FieldParser<::imex::xegpu::MemoryScope,
                              ::imex::xegpu::MemoryScope>::parse(parser);
      if (mlir::failed(attrOptional))
        return parser.emitError(
            loc, "Invalid memory scope attribute specification.\n");
      scope = *attrOptional;
      return mlir::success();
    } else {
      auto loc = parser.getCurrentLocation();
      auto attrOptional = ::mlir::FieldParser<::mlir::Attribute>::parse(parser);
      if (mlir::failed(attrOptional))
        return parser.emitError(
            loc, "Failed to parse XeGPU_TensorDesc parameter 'encoding' which "
                 "is to be a `::mlir::Attribute`.\n");

      if (llvm::isa<imex::xegpu::ScatteredAttr>(*attrOptional))
        encoding = *attrOptional;

      if (llvm::isa<imex::xegpu::SubGroupMapAttr>(*attrOptional) ||
          llvm::isa<imex::xegpu::WorkGroupMapAttr>(*attrOptional) ||
          llvm::isa<imex::xegpu::XeMapAttr>(*attrOptional))
        mapping = *attrOptional;
      return mlir::success();
    }
  };

  if (parser.parseCommaSeparatedList(parseElt))
    return mlir::failure();

  return mlir::success();
}

// custom printer for XeGPU_TensorDesc (scope, encoding and mapping parameter)
static void printTensorDescAttr(mlir::AsmPrinter &printer,
                                imex::xegpu::MemoryScope scope,
                                mlir::Attribute encoding,
                                mlir::Attribute mapping) {
  if (printDefaultValues() || scope != imex::xegpu::MemoryScope::GLOBAL)
    printer << ", memory_scope = " << scope;
  if (encoding)
    printer << ", " << encoding;
  if (mapping)
    printer << ", " << mapping;
}

mlir::LogicalResult SubGroupMapAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::DenseI32ArrayAttr mmaBlockSize, mlir::DenseI32ArrayAttr layout,
    mlir::DenseI32ArrayAttr data) {

  if (layout.size() != 2) {
    emitError() << "Failed to parse SubGroupMapAttr: missing wi_layout which "
                   "is to be an integer array of size 2.\n";
    return mlir::failure();
  }

  if (data.size() != 2) {
    emitError() << "Failed to parse SubGroupMapAttr: missing wi_data which is "
                   "to be an integer array of size 2.\n";
    return mlir::failure();
  }

  if (mmaBlockSize) {
    if (mmaBlockSize.size() != 2) {
      emitError()
          << "Failed to parse SubGroupMapAttr: the optional mma_block_size "
             "should be an integer array of size 2 or empty. But it got "
          << mmaBlockSize.size() << ".\n";
      return mlir::failure();
    }
    for (int i = 0; i < mmaBlockSize.size(); i++) {
      if ((mmaBlockSize[i] % (layout[i] * data[i]) != 0 &&
           (layout[i] * data[i]) % mmaBlockSize[i] != 0) ||
          mmaBlockSize[i] % layout[i] != 0 || mmaBlockSize[i] % data[i] != 0) {
        return emitError()
               << "Invalid SubGroupMapAttr. A valid SubGroupMapAttr should "
                  "meet the following conditions: "
                  "\n\tmmaBlockSize[i] % wi_layout[i] == 0 && "
                  "\n\tmmaBlockSize[i] % wi_data[i] == 0 && "
                  "\n\t(mmaBlockSize[i] % (wi_layout[i] * wi_data[i]) == 0 || "
                  "\n\t (wi_layout[i] * wi_data[i]) % mmaBlockSize[i] == 0)";
      }
    }
  }

  return mlir::success();
}

mlir::LogicalResult WorkGroupMapAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::DenseI32ArrayAttr layout, mlir::DenseI32ArrayAttr data) {

  if (layout.size() != 2) {
    emitError() << "Failed to parse WorkGroupMapAttr: missing sg_layout which "
                   "is to be a `llvm::ArrayRef<int32_t>` with size 2.\n";
    return mlir::failure();
  }
  if (data.size() != 2) {
    emitError() << "Failed to parse WorkGroupMapAttr: missing sg_data which is "
                   "to be a `llvm::ArrayRef<int32_t>` with size 2.\n";
    return mlir::failure();
  }
  return mlir::success();
}

} // namespace xegpu
} // namespace imex

#include "TPP/Dialect/XeGPU/IR/XeGPUOpsDialect.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "TPP/Dialect/XeGPU/IR/XeGPUOpsAttrs.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "TPP/Dialect/XeGPU/IR/XeGPUOpsTypes.cpp.inc"
