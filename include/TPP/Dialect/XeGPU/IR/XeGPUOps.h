//===- XeGPUOps.h - XeGPU dialect  -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the XeGPU dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _XeGPU_OPS_H_INCLUDED_
#define _XeGPU_OPS_H_INCLUDED_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/CopyOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/ShapedOpInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace mlir {

/// Return the list of Range (i.e. offset, size, stride). Each Range
/// entry contains either the dynamic value or a ConstantIndexOp constructed
/// with `b` at location `loc`.
SmallVector<Range, 8> getOrCreateRanges(OffsetSizeAndStrideOpInterface op,
                                        OpBuilder &b, Location loc);

namespace xegpu {

class TileType;

} // namespace xegpu

namespace xegpu {

class TileBase : public mlir::Type, public mlir::ShapedType::Trait<TileBase> {
public:
  using Type::Type;

  /// Returns the element type of this tensor type.
  mlir::Type getElementType() const;

  /// Returns if this type is ranked, i.e. it has a known number of dimensions.
  bool hasRank() const;

  /// Returns the shape of this tensor type.
  llvm::ArrayRef<int64_t> getShape() const;

  /// Clone this type with the given shape and element type. If the
  /// provided shape is `None`, the current shape of the type is used.
  TileBase cloneWith(std::optional<llvm::ArrayRef<int64_t>> shape,
                     mlir::Type elementType) const;

  /// Return true if the specified element type is ok in a tensor.
  static bool isValidElementType(Type type);

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(Type type);

  /// Allow implicit conversion to ShapedType.
  operator mlir::ShapedType() const { return cast<mlir::ShapedType>(); }
};

} // namespace xegpu
} // namespace mlir

#include "TPP/Dialect/XeGPU/IR/XeGPUOpsDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "TPP/Dialect/XeGPU/IR/XeGPUOpsTypes.h.inc"
#define GET_OP_CLASSES
#include "TPP/Dialect/XeGPU/IR/XeGPUOps.h.inc"

#endif // _XeGPU_OPS_H_INCLUDED_
