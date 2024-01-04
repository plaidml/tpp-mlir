//===- TppUtils.cpp ----------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Tpp/TppUtils.h"
#include "TPP/Dialect/Tpp/TppOps.h"
#include "TPP/IR/StructuredOpMatcher.h"
#include "TPP/Transforms/Utils/VNNIUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
namespace tpp {
namespace utils {

LogicalResult splitAndReplaceFusedOp(tpp::FusedBrgemmOp fusedBrgemmOp,
                                     PatternRewriter &rewriter) {
  if (!fusedBrgemmOp.hasBufferSemantics())
    return failure();

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(fusedBrgemmOp);

  Location loc = fusedBrgemmOp.getLoc();

  // Split the fused op into individual operations.
  auto ins = fusedBrgemmOp.getInputs();
  auto out = fusedBrgemmOp.getOutput();
  rewriter.create<tpp::BrgemmOp>(loc, ValueRange{ins[0], ins[1], ins[2]}, out);

  switch (fusedBrgemmOp.getBinaryKind()) {
  case tpp::FusedBinaryOpKind::ADD:
    rewriter.create<tpp::AddOp>(loc, ValueRange{ins[3], out}, out);
    break;
  case tpp::FusedBinaryOpKind::NONE:
    break;
  }

  switch (fusedBrgemmOp.getUnaryKind()) {
  case tpp::FusedUnaryOpKind::RELU:
    rewriter.create<tpp::ReluOp>(loc, out, out);
    break;
  case tpp::FusedUnaryOpKind::NONE:
    break;
  }

  rewriter.eraseOp(fusedBrgemmOp);
  return success();
}

} // namespace utils
} // namespace tpp
} // namespace mlir
