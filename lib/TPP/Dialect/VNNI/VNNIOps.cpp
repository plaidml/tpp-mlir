//===- VNNIOps.cpp - VNNI Ops implementation --------------------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/VNNI/VNNIOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/OpImplementation.h"
#define GET_OP_CLASSES
#include "TPP/Dialect/VNNI/VNNIOps.cpp.inc"

using namespace mlir;
using namespace mlir::vnni;

void MatmulOp::build(OpBuilder &builder, OperationState &state, Value matrixA,
                     Value matrixB, Value matrixC) {
  MatmulOp::build(builder, state, TypeRange{}, matrixA, matrixB, matrixC);
}
