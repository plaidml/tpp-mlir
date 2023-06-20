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
    MatchSelector range,
    std::function<bool(OpOperand *operand, Operation *op)> fun) {
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
    auto operands = linalgOp.getDpsInputOperands();
    size_t upperBound = range.getUpperBound();
    size_t lowerBound = range.getLowerBound();
    if (upperBound == std::numeric_limits<size_t>::max())
      upperBound = operands.size();

    for (auto idx :
         llvm::to_vector(llvm::seq<size_t>(lowerBound, upperBound))) {
      if (!fun(operands[idx], linalgOp.getOperation()))
        return false;
    }
    return true;
  });
  return *this;
}

//===---------------------------------------------------------------------===//
// Operand predicates - output.
//===---------------------------------------------------------------------===//

structured_match::StructuredOpMatcher &
structured_match::StructuredOpMatcher::output(
    MatchSelector range,
    std::function<bool(OpOperand *operand, Operation *operation)> fun) {
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
    auto operands = linalgOp.getDpsInitOperands();
    size_t upperBound = range.getUpperBound();
    size_t lowerBound = range.getLowerBound();
    if (upperBound == std::numeric_limits<size_t>::max())
      upperBound = operands.size();

    for (auto idx :
         llvm::to_vector(llvm::seq<size_t>(lowerBound, upperBound))) {
      if (!fun(operands[idx], linalgOp.getOperation()))
        return false;
    }
    return true;
  });
  return *this;
}

//===---------------------------------------------------------------------===//
// Dim predicates.
//===---------------------------------------------------------------------===//

structured_match::StructuredOpMatcher &
structured_match::StructuredOpMatcher::dim(
    MatchSelector range, SmallVector<utils::IteratorType> kinds) {
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
structured_match::StructuredOpMatcher::dim(MatchSelector range,
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
    StringRef operationName, Region *region, Operation *op,
    SmallVectorImpl<Value> *capturedOperands) {
  if (!isa<linalg::LinalgOp>(op))
    return false;
  auto linalgOp = cast<linalg::LinalgOp>(op);

  if (!region->hasOneBlock())
    return false;
  unsigned numberOfOpsInRegion =
      (operationName.compare(linalg::YieldOp::getOperationName()) == 0) ? 1 : 2;
  if (std::distance(region->front().begin(), region->front().end()) !=
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
      // linalg.yield operand might be coming from a different region.
      if (arg0 && arg0.getParentBlock() == linalgOp.getBlock())
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

// FIXME: This is a generalization of the method above and will eventually
// replace the matcher for both no-op (yield) and one op (add, max).
bool tpp::structured_match::withOpChainImpl(
    Region *region, Operation *op, SmallVectorImpl<Value> *capturedOperands,
    SmallVectorImpl<TypeCheckFunc> &typeChecks) {

  // Number of ops includes yield
  ptrdiff_t numOps = typeChecks.size() + 1;

  // Basic checks
  if (!isa<linalg::GenericOp>(op))
    return false;
  auto linalgOp = cast<linalg::GenericOp>(op);
  if (!region->hasOneBlock())
    return false;
  auto &block = region->front();
  if (std::distance(block.begin(), block.end()) != numOps)
    return false;
  if (linalgOp.getNumDpsInits() != 1)
    return false;

  // Add generic arguments to the list of chained values
  llvm::SmallSetVector<Value, 4> chainedValues;
  for (auto arg : block.getArguments()) {
    chainedValues.insert(arg);
  }

  // Check on the inner chain of operations in the right order.
  // Make sure all operands are used and chained
  for (auto [check, innerOp] :
       llvm::zip_first(typeChecks, block.getOperations())) {
    // Must be right op in right order
    if (!check(&innerOp))
      return false;

    // At least one operand must come from args or a previous op
    bool consumesValueFromChain = false;
    for (auto operand : innerOp.getOperands()) {
      if (chainedValues.contains(operand) && capturedOperands) {
        // First add to the captured
        auto ba = dyn_cast<BlockArgument>(operand);
        if (ba && ba.getParentBlock() == linalgOp.getBlock()) {
          capturedOperands->push_back(linalgOp.getMatchingOpOperand(ba)->get());
        }
      }
      // Then erase from the set
      chainedValues.remove(operand);
      consumesValueFromChain = true;
    }

    // Operation isn't in the chain
    if (!consumesValueFromChain)
      return false;

    // Add return value to the list of chained values
    for (auto ret : innerOp.getResults()) {
      chainedValues.insert(ret);
    }
  }

  // Last op must be a chained yield.
  Operation *yieldOp = linalgOp.getBlock()->getTerminator();
  assert(isa<linalg::YieldOp>(yieldOp) && "Wrong terminator");
  for (auto op : yieldOp->getOperands()) {
    if (!chainedValues.contains(op))
      return false;
  }

  return true;
}

structured_match::StructuredOpMatcher &
structured_match::StructuredOpMatcher::region(
    MatchSelector range,
    std::function<bool(Region *region, Operation *op)> fun) {
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
    auto regions = linalgOp->getRegions();
    assert(regions.size() != 0);
    size_t upperBound = range.getUpperBound();
    size_t lowerBound = range.getLowerBound();
    if (upperBound == std::numeric_limits<size_t>::max())
      upperBound = regions.size();

    for (auto idx :
         llvm::to_vector(llvm::seq<size_t>(lowerBound, upperBound))) {
      if (!fun(&regions[idx], linalgOp.getOperation()))
        return false;
    }
    return true;
  });
  return *this;
}
