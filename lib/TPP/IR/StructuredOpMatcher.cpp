//===- StructuredOpMatcher.cpp -----------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/IR/StructuredOpMatcher.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "structured-matchers"

using namespace mlir;
using namespace mlir::tpp;

// Entry point.
bool structured_match::StructuredOpMatcher::match(Operation *op) {
  auto linalgOp = dyn_cast_or_null<linalg::LinalgOp>(op);
  if (!linalgOp)
    return false;
  LLVM_DEBUG(llvm::dbgs() << "Running matcher on: " << *op << "\n");

  for (auto [idx, predicate] : llvm::enumerate(predicates)) {
    if (!predicate(linalgOp)) {
      LLVM_DEBUG(llvm::dbgs() << "Exit on predicate: " << idx << "\n");
      return false;
    }
  }
  return true;
}

//===---------------------------------------------------------------------===//
// Operation predicates.
//===---------------------------------------------------------------------===//

structured_match::StructuredOpMatcher &
structured_match::StructuredOpMatcher::operation(
    std::function<bool(Operation *op)> fun) {
  predicates.push_back(
      [=](linalg::LinalgOp linalgOp) -> bool { return fun(linalgOp); });
  return *this;
}

//===---------------------------------------------------------------------===//
// Operand predicates - input.
//===---------------------------------------------------------------------===//

structured_match::StructuredOpMatcher &
structured_match::StructuredOpMatcher::input(
    AllOperands tag,
    std::function<bool(OpOperand *operand, Operation *op)> fun) {
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
    for (OpOperand *operand : linalgOp.getDpsInputOperands()) {
      if (!fun(operand, linalgOp.getOperation()))
        return false;
    }
    return true;
  });
  return *this;
}

structured_match::StructuredOpMatcher &
structured_match::StructuredOpMatcher::input(
    Operand operand,
    std::function<bool(OpOperand *operand, Operation *op)> fun) {
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
    size_t idxOperand = operand.idx;
    assert(idxOperand < static_cast<size_t>(linalgOp.getNumDpsInputs()));
    return fun(linalgOp.getDpsInputOperand(idxOperand),
               linalgOp.getOperation());
  });
  return *this;
}

//===---------------------------------------------------------------------===//
// Operand predicates - output.
//===---------------------------------------------------------------------===//

structured_match::StructuredOpMatcher &
structured_match::StructuredOpMatcher::output(
    AllOperands tag,
    std::function<bool(OpOperand *operand, Operation *operation)> fun) {
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
    for (OpOperand *operand : linalgOp.getDpsInitOperands()) {
      if (!fun(operand, linalgOp.getOperation()))
        return false;
    }
    return true;
  });
  return *this;
}

structured_match::StructuredOpMatcher &
structured_match::StructuredOpMatcher::output(
    Operand operand,
    std::function<bool(OpOperand *operand, Operation *operation)> fun) {
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
    size_t idxOperand = operand.idx;
    assert(idxOperand < static_cast<size_t>(linalgOp.getNumDpsInits()));
    return fun(linalgOp.getDpsInitOperand(idxOperand), linalgOp.getOperation());
  });
  return *this;
}

//===---------------------------------------------------------------------===//
// Dim predicates.
//===---------------------------------------------------------------------===//

structured_match::StructuredOpMatcher &
structured_match::StructuredOpMatcher::dim(
    RangeDims range, SmallVector<utils::IteratorType> kinds) {
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
    size_t upperBound = range.getUpperBound();
    size_t lowerBound = range.getLowerBound();
    if (upperBound == std::numeric_limits<size_t>::max())
      upperBound = kinds.size();
    size_t sizeRange = upperBound - lowerBound;

    auto iteratorTypes = linalgOp.getIteratorTypesArray();
    if (iteratorTypes.size() != sizeRange)
      return false;
    // Reverse iterators to have the innermost one at index 0.
    std::reverse(iteratorTypes.begin(), iteratorTypes.end());
    for (auto [idx, rangeIdx] :
         llvm::enumerate(llvm::seq<size_t>(lowerBound, upperBound))) {
      if (iteratorTypes[rangeIdx] != kinds[idx])
        return false;
    }
    return true;
  });
  return *this;
}

structured_match::StructuredOpMatcher &
structured_match::StructuredOpMatcher::dim(RangeDims range,
                                           utils::IteratorType kind) {
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
    auto iteratorTypes = linalgOp.getIteratorTypesArray();
    size_t upperBound = range.getUpperBound();
    size_t lowerBound = range.getLowerBound();
    if (upperBound == std::numeric_limits<size_t>::max())
      upperBound = iteratorTypes.size();

    for (auto rangeIdx = lowerBound; rangeIdx < upperBound; rangeIdx++) {
      if (iteratorTypes[rangeIdx] != kind)
        return false;
    }
    return true;
  });
  return *this;
}

//===---------------------------------------------------------------------===//
// Region predicates.
//===---------------------------------------------------------------------===//

bool tpp::structured_match::WithSingleOpImpl::withSingleOpImpl(
    StringRef operationName, Operation *op,
    SmallVectorImpl<Value> *capturedOperands) {
  if (!isa<linalg::LinalgOp>(op))
    return false;
  auto linalgOp = cast<linalg::LinalgOp>(op);
  Region &region = linalgOp->getRegion(0);

  if (!region.hasOneBlock())
    return false;
  unsigned numberOfOpsInRegion =
      (operationName.compare(linalg::YieldOp::getOperationName()) == 0) ? 1 : 2;
  if (std::distance(region.front().begin(), region.front().end()) !=
      numberOfOpsInRegion)
    return false;
  if (linalgOp.getNumDpsInits() != 1)
    return false;

  // Require only a single yield operand defined by innerOp.
  Operation *yieldOp = linalgOp.getBlock()->getTerminator();
  if (yieldOp->getNumOperands() != 1)
    return false;
  // Only linalg.yield, exit true.
  if (numberOfOpsInRegion == 1) {
    if (capturedOperands) {
      auto arg0 = dyn_cast<BlockArgument>(yieldOp->getOperand(0));
      if (!arg0 || arg0.getParentBlock() != linalgOp.getBlock())
        return false;
      capturedOperands->push_back(linalgOp.getMatchingOpOperand(arg0)->get());
      capturedOperands->push_back(linalgOp.getDpsInitOperand(0)->get());
    }
    return true;
  }

  // Check on the only inner operation.
  Operation *innerOp = &(*linalgOp.getBlock()->getOperations().begin());
  if (innerOp->getName().getStringRef() != operationName)
    return false;
  if (yieldOp->getOperand(0).getDefiningOp() != innerOp)
    return false;
  // The operand of the innerOp must comes from the region
  // args of the generic.
  auto arg0 = dyn_cast<BlockArgument>(innerOp->getOperand(0));
  auto arg1 = dyn_cast<BlockArgument>(innerOp->getOperand(1));
  if (!arg0 || !arg1)
    return false;
  if (arg0.getParentBlock() != linalgOp.getBlock() ||
      arg1.getParentBlock() != linalgOp.getBlock())
    return false;
  if (capturedOperands) {
    capturedOperands->push_back(linalgOp.getMatchingOpOperand(arg0)->get());
    capturedOperands->push_back(linalgOp.getMatchingOpOperand(arg1)->get());
    capturedOperands->push_back(linalgOp.getDpsInitOperand(0)->get());
  }
  return true;
}

structured_match::StructuredOpMatcher &
structured_match::StructuredOpMatcher::region(
    std::function<bool(Operation *op, SmallVectorImpl<Value> *capturedOperands)>
        fun,
    SmallVectorImpl<Value> *capturedOperands) {
  predicates.push_back([=](linalg::LinalgOp linalgOp) {
    return fun(linalgOp, capturedOperands);
  });
  return *this;
}
