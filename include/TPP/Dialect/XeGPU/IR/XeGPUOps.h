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

} // namespace mlir

namespace imex {
namespace xegpu {

class TensorDescType;

} // namespace xegpu
} // namespace imex

#include "TPP/Dialect/XeGPU/IR/XeGPUOpsDialect.h.inc"
#include "TPP/Dialect/XeGPU/IR/XeGPUOpsEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "TPP/Dialect/XeGPU/IR/XeGPUOpsAttrs.h.inc"
#define GET_TYPEDEF_CLASSES
#include "TPP/Dialect/XeGPU/IR/XeGPUOpsTypes.h.inc"
#define GET_OP_CLASSES
#include "TPP/Dialect/XeGPU/IR/XeGPUOps.h.inc"

#endif // _XeGPU_OPS_H_INCLUDED_
