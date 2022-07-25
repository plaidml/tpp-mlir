//===- LinalgXOps.cpp - LinalgX dialect ops ---------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/LinalgX/LinalgXOps.h"
#include "Standalone/Dialect/LinalgX/LinalgXDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::linalgx;

//===----------------------------------------------------------------------===//
// ToBlockLayout
//===----------------------------------------------------------------------===//

ArrayAttr ToBlockLayout::getIndexingMaps() {
  MLIRContext *context = getContext();
  auto maybeInputMap = getInputMap();
  auto maybeOutputMap = getOutputMap();
  int64_t inputRank = getRank(getInputOperand(0));
  int64_t outputRank = getRank(getOutputOperand(0));
  return Builder(getContext())
      .getAffineMapArrayAttr(
          {linalg::extractOrIdentityMap(maybeInputMap, inputRank, context),
           linalg::extractOrIdentityMap(maybeOutputMap, outputRank, context)});
}

ArrayAttr ToBlockLayout::iterator_types() {
  int64_t numLoops = getTiedIndexingMap(getInputOperand(0)).getNumDims();
  return Builder(getContext())
      .getStrArrayAttr(
          SmallVector<StringRef, 8>(numLoops, getParallelIteratorTypeName()));
}

std::string ToBlockLayout::getLibraryCallName() { return "to_block_layout"; }

void ToBlockLayout::regionBuilder(ImplicitLocOpBuilder &b, Block &block,
                                  llvm::ArrayRef<NamedAttribute> attrs = {}) {
  assert(block.getNumArguments() == 2 && "CopyOp regionBuilder expects 2 args");
  b.create<linalg::YieldOp>(block.getArgument(0));
}

//===----------------------------------------------------------------------===//
// FromBlockLayout
//===----------------------------------------------------------------------===//

ArrayAttr FromBlockLayout::getIndexingMaps() {
  MLIRContext *context = getContext();
  auto maybeInputMap = getInputMap();
  auto maybeOutputMap = getOutputMap();
  int64_t inputRank = getRank(getInputOperand(0));
  int64_t outputRank = getRank(getOutputOperand(0));
  return Builder(getContext())
      .getAffineMapArrayAttr(
          {linalg::extractOrIdentityMap(maybeInputMap, inputRank, context),
           linalg::extractOrIdentityMap(maybeOutputMap, outputRank, context)});
}

ArrayAttr FromBlockLayout::iterator_types() {
  int64_t numLoops = getTiedIndexingMap(getInputOperand(0)).getNumDims();
  return Builder(getContext())
      .getStrArrayAttr(
          SmallVector<StringRef, 8>(numLoops, getParallelIteratorTypeName()));
}

std::string FromBlockLayout::getLibraryCallName() {
  return "from_block_layout";
}

void FromBlockLayout::regionBuilder(ImplicitLocOpBuilder &b, Block &block,
                                    llvm::ArrayRef<NamedAttribute> attrs = {}) {
  assert(block.getNumArguments() == 2 && "CopyOp regionBuilder expects 2 args");
  b.create<linalg::YieldOp>(block.getArgument(0));
}

#define GET_OP_CLASSES
#include "Standalone/Dialect/LinalgX/LinalgXOps.cpp.inc"
