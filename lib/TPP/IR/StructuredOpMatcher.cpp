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
structured_match::StructuredOpMatcher::outputs(
    std::function<bool(size_t)> fun) {
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
    size_t numDpsInits = static_cast<size_t>(linalgOp.getNumDpsInits());
    return fun(numDpsInits);
  });
  return *this;
}

structured_match::StructuredOpMatcher &
structured_match::StructuredOpMatcher::inputs(std::function<bool(size_t)> fun) {
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
    size_t numDpsInputs = static_cast<size_t>(linalgOp.getNumDpsInputs());
    return fun(numDpsInputs);
  });
  return *this;
}

structured_match::StructuredOpMatcher &
structured_match::StructuredOpMatcher::inputs(
    structured_match::BinaryPredicate binaryPredicate) {
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
    if (isa<_OR>(&binaryPredicate)) {
      return (binaryPredicate.predicateOnLhs(linalgOp.getNumDpsInputs()) ||
              binaryPredicate.predicateOnRhs(linalgOp.getNumDpsInputs()));
    }
    return false;
  });
  return *this;
}

structured_match::StructuredOpMatcher &
structured_match::StructuredOpMatcher::hasBufferSemantics() {
  predicates.push_back([](linalg::LinalgOp linalgOp) -> bool {
    return linalgOp.hasBufferSemantics();
  });
  return *this;
}

structured_match::StructuredOpMatcher &
structured_match::StructuredOpMatcher::hasTensorSemantics() {
  predicates.push_back([](linalg::LinalgOp linalgOp) -> bool {
    return linalgOp.hasTensorSemantics();
  });
  return *this;
}

structured_match::StructuredOpMatcher &
structured_match::StructuredOpMatcher::verifyInterface(
    std::function<LogicalResult(Operation *op)> fun) {
  predicates.push_back([&](linalg::LinalgOp linalgOp) -> bool {
    if (failed(fun(linalgOp)))
      return false;
    return true;
  });
  return *this;
}

//===---------------------------------------------------------------------===//
// Input predicates.
//===---------------------------------------------------------------------===//

structured_match::StructuredOpMatcher &
structured_match::StructuredOpMatcher::input(AllOperands tag, HasStaticShape) {
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
    for (OpOperand *operand : linalgOp.getDpsInputOperands()) {
      auto operandType = operand->get().getType();
      if (auto shapedType = operandType.dyn_cast_or_null<ShapedType>())
        if (!shapedType.hasStaticShape())
          return false;
    }
    return true;
  });
  return *this;
}

structured_match::StructuredOpMatcher &
structured_match::StructuredOpMatcher::input(
    Operand operand, std::function<bool(AffineMap map)> fun) {
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
    size_t idxOperand = operand.idx;
    assert(idxOperand < static_cast<size_t>(linalgOp.getNumDpsInputs()));
    AffineMap tiedIndexingMap = linalgOp.getMatchingIndexingMap(
        linalgOp.getDpsInputOperand(idxOperand));
    return fun(tiedIndexingMap);
  });
  return *this;
}

structured_match::StructuredOpMatcher &
structured_match::StructuredOpMatcher::input(
    AllOperands tag, std::function<bool(AffineMap map)> fun) {
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
    for (OpOperand *operand : linalgOp.getDpsInputOperands()) {
      AffineMap tiedIndexingMap = linalgOp.getMatchingIndexingMap(operand);
      if (!fun(tiedIndexingMap))
        return false;
    }
    return true;
  });
  return *this;
}

//===---------------------------------------------------------------------===//
// Output predicates.
//===---------------------------------------------------------------------===//

structured_match::StructuredOpMatcher &
structured_match::StructuredOpMatcher::output(AllOperands tag, HasStaticShape) {
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
    for (OpOperand *operand : linalgOp.getDpsInitOperands()) {
      auto operandType = operand->get().getType();
      if (auto shapedType = operandType.dyn_cast_or_null<ShapedType>())
        if (!shapedType.hasStaticShape())
          return false;
    }
    return true;
  });
  return *this;
}

structured_match::StructuredOpMatcher &
structured_match::StructuredOpMatcher::output(
    Operand operand, std::function<bool(AffineMap map)> fun) {
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
    size_t idxOperand = operand.idx;
    assert(idxOperand < static_cast<size_t>(linalgOp.getNumDpsInits()));
    AffineMap tiedIndexingMap =
        linalgOp.getMatchingIndexingMap(linalgOp.getDpsInitOperand(idxOperand));
    return fun(tiedIndexingMap);
  });
  return *this;
}

structured_match::StructuredOpMatcher &
structured_match::StructuredOpMatcher::output(
    AllOperands tag, std::function<bool(AffineMap map)> fun) {
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
    for (OpOperand *operand : linalgOp.getDpsInitOperands()) {
      AffineMap tiedIndexingMap = linalgOp.getMatchingIndexingMap(operand);
      if (!fun(tiedIndexingMap))
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
structured_match::StructuredOpMatcher::dim(std::function<bool(size_t)> fun) {
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
    size_t numberOfIterator = linalgOp.getIteratorTypesArray().size();
    return fun(numberOfIterator);
  });
  return *this;
}

structured_match::StructuredOpMatcher &
structured_match::StructuredOpMatcher::dim(AllDims tag,
                                           utils::IteratorType kind) {
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
    auto iteratorTypes = linalgOp.getIteratorTypesArray();
    for (auto iteratorType : iteratorTypes)
      if (iteratorType != kind)
        return false;
    return true;
  });
  return *this;
}

structured_match::StructuredOpMatcher &
structured_match::StructuredOpMatcher::dim(
    AllDims tag, SmallVector<utils::IteratorType> kinds) {
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
    auto iteratorTypes = linalgOp.getIteratorTypesArray();
    return iteratorTypes == kinds;
  });
  return *this;
}

structured_match::StructuredOpMatcher &
structured_match::StructuredOpMatcher::dim(
    RangeDims range, SmallVector<utils::IteratorType> kinds) {
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
    if (kinds.size() != range.getSize())
      return false;
    auto iteratorTypes = linalgOp.getIteratorTypesArray();
    // Reverse iterators to have the innermost one at index 0.
    std::reverse(iteratorTypes.begin(), iteratorTypes.end());
    for (auto [idx, rangeIdx] : llvm::enumerate(range.getRange())) {
      if (iteratorTypes[rangeIdx] != kinds[idx])
        return false;
    }
    return true;
  });
  return *this;
}

structured_match::StructuredOpMatcher &
structured_match::StructuredOpMatcher::dim(AllDimsBut allDimsBut,
                                           utils::IteratorType kind) {
  predicates.push_back([=](linalg::LinalgOp linalgOp) -> bool {
    llvm::DenseSet<size_t> exceptions(allDimsBut.getExceptions().begin(),
                                      allDimsBut.getExceptions().end());
    auto iteratorTypes = linalgOp.getIteratorTypesArray();
    // Reverse iterators to have the innermost one at index 0.
    std::reverse(iteratorTypes.begin(), iteratorTypes.end());
    for (auto [idx, iteratorType] : llvm::enumerate(iteratorTypes)) {
      if (exceptions.contains(idx))
        continue;
      if (iteratorType != kind)
        return false;
    }
    return true;
  });
  return *this;
}

//===---------------------------------------------------------------------===//
// Region predicates.
//===---------------------------------------------------------------------===//

structured_match::StructuredOpMatcher &
structured_match::StructuredOpMatcher::hasRegionWithSingleOpImpl(
    StringRef operationName, SmallVectorImpl<Value> *capturedOperands) {
  predicates.push_back([=](linalg::LinalgOp linalgOp) {
    Region &region = linalgOp->getRegion(0);

    if (!region.hasOneBlock())
      return false;
    unsigned numberOfOpsInRegion =
        (operationName.compare("linalg.yield") == 0) ? 1 : 2;
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
  });
  return *this;
}

structured_match::StructuredOpMatcher &
structured_match::StructuredOpMatcher::hasRegion(
    std::function<bool(Operation *op, SmallVectorImpl<Value> *capturedOperands)>
        fun,
    SmallVectorImpl<Value> *capturedOperands) {
  predicates.push_back([=](linalg::LinalgOp linalgOp) {
    return fun(linalgOp, capturedOperands);
  });
  return *this;
}
