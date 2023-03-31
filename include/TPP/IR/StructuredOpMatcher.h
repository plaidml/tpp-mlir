//===- StructuredOpMatcher.h -------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_STRUCTUREDOPMATCHERS_H
#define TPP_STRUCTUREDOPMATCHERS_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/ADT/SmallVector.h"
#include <functional>

namespace mlir {
class Operation;
namespace tpp {
namespace structured_match {

// Base class for the matcher predicates selection tag.
struct MatchSelector {
  MatchSelector() = delete;
  size_t getLowerBound() const { return lowerBound; };
  size_t getUpperBound() const { return upperBound; };

protected:
  explicit MatchSelector(size_t lowerBound, size_t upperBound)
      : lowerBound(lowerBound), upperBound(upperBound) {
    assert(upperBound > lowerBound);
  }

private:
  const size_t lowerBound;
  const size_t upperBound;
};

// Selector which specifies that predicate should apply on all values.
struct MatchAll : public MatchSelector {
  MatchAll() : MatchSelector(0, std::numeric_limits<size_t>::max()) {}
};

// Selector which specifies that predicate should apply only on one value at
// the position `idx`.
struct MatchOne : public MatchSelector {
  MatchOne() = delete;
  MatchOne(size_t idx) : MatchSelector(idx, idx + 1) {}
};

// Selector which specifies that predicate should apply only on range of values
// at positions from `lowerBound` up to - but not including - `upperBound`.
struct MatchRange : public MatchSelector {
  MatchRange() = delete;
  MatchRange(size_t lowerBound, size_t upperBound)
      : MatchSelector(lowerBound, upperBound) {}
};

// Callable object to check if the number of loops in `op` satisfies `fun`.
struct NumOfLoops {
  NumOfLoops() = delete;
  NumOfLoops(std::function<bool(size_t)> fun) : fun(fun){};

  bool operator()(Operation *op) const {
    if (auto linalgOp = dyn_cast_or_null<linalg::LinalgOp>(op)) {
      auto numberOfLoops = linalgOp.getNumLoops();
      return fun(numberOfLoops);
    }
    return false;
  }
  std::function<bool(size_t)> fun;
};

// Callable object to check if the `operand` of `op` has a map that satisfies
// `fun`.
struct HasMap {
  HasMap() = delete;
  HasMap(std::function<bool(AffineMap)> fun) : fun(fun){};

  bool operator()(OpOperand *operand, Operation *op) const {
    if (auto linalgOp = dyn_cast_or_null<linalg::LinalgOp>(op)) {
      auto map = linalgOp.getMatchingIndexingMap(operand);
      return fun(map);
    }
    return false;
  }
  std::function<bool(AffineMap map)> fun;
};

// Callble object to verify if `map` is a projected permutation map.
struct ProjectedPermutation {
  ProjectedPermutation() = default;

  bool operator()(AffineMap map) const {
    return map.isProjectedPermutation(/*allowZeroInResults=*/true);
  }
};

// Callable object to verify if `map` is an identity map.
struct Identity {
  Identity() = default;

  bool operator()(AffineMap map) const { return map.isIdentity(); }
};

// Callable object to verify if `operand` has static shape.
struct HasStaticShape {
  HasStaticShape() = default;

  bool operator()(OpOperand *operand, Operation *op) const {
    auto operandType = operand->get().getType();
    if (auto shapedType = operandType.dyn_cast_or_null<ShapedType>())
      if (!shapedType.hasStaticShape())
        return false;
    return true;
  }
};

// Callable object to check if the input is equal to specified `value`.
template <typename T> struct EqualsTo {
  EqualsTo() = delete;
  explicit EqualsTo(T value) : value(value){};

  const T value;

  bool operator()(T value) const { return value == this->value; }
};
template <typename T> EqualsTo(T) -> EqualsTo<T>;

// Callable object to check if the input is less than or equal to specified
// `value`.
struct LessThanOrEqualTo {
  LessThanOrEqualTo() = delete;
  explicit LessThanOrEqualTo(size_t value) : value(value){};
  const size_t value;

  bool operator()(size_t value) const { return value <= this->value; }
};

// Callable object to check if the input is greater than or equal to specified
// `value`.
struct GreaterThanOrEqualTo {
  GreaterThanOrEqualTo() = delete;
  explicit GreaterThanOrEqualTo(size_t value) : value(value){};
  const size_t value;

  bool operator()(size_t value) const { return value >= this->value; }
};

// Callable object to check if `op` has tensor semantics.
struct HasTensorSemantics {
  HasTensorSemantics() = default;

  bool operator()(Operation *op) const {
    if (auto linalgOp = dyn_cast_or_null<linalg::LinalgOp>(op))
      return linalgOp.hasTensorSemantics();
    return false;
  }
};

// Callable object to check if `op` buffer semantics.
struct HasBufferSemantics {
  HasBufferSemantics() = default;

  bool operator()(Operation *op) const {
    if (auto linalgOp = dyn_cast_or_null<linalg::LinalgOp>(op))
      return linalgOp.hasBufferSemantics();
    return false;
  }
};

// Callable object to validate number of init operands for `op`.
struct NumDpsInits {
  NumDpsInits() = delete;
  explicit NumDpsInits(std::function<bool(size_t)> fun) : fun(fun){};

  bool operator()(Operation *op) const {
    if (auto linalgOp = dyn_cast_or_null<linalg::LinalgOp>(op))
      return fun(linalgOp.getNumDpsInits());
    return false;
  }

  std::function<bool(size_t)> fun;
};

// Callable object to validate number of input operands for `op`.
struct NumDpsInputs {
  NumDpsInputs() = delete;
  explicit NumDpsInputs(std::function<bool(size_t)> fun) : fun(fun){};

  bool operator()(Operation *op) {
    if (auto linalgOp = dyn_cast_or_null<linalg::LinalgOp>(op))
      return fun(linalgOp.getNumDpsInputs());
    return false;
  }

  std::function<bool(size_t)> fun;
};

// Callable object to validate number of regions for `op`.
struct NumRegions {
  NumRegions() = delete;
  explicit NumRegions(std::function<bool(size_t)> fun) : fun(fun){};

  bool operator()(Operation *op) const {
    if (auto linalgOp = dyn_cast_or_null<linalg::LinalgOp>(op))
      return fun(linalgOp->getNumRegions());
    return false;
  }

  std::function<bool(size_t)> fun;
};

// Logical OR between two predicates.
struct _OR {
  _OR() = delete;
  _OR(std::function<bool(size_t)> lhs, std::function<bool(size_t)> rhs)
      : lhs(lhs), rhs(rhs) {}

  bool operator()(size_t num) { return (lhs(num) || rhs(num)); }

  std::function<bool(size_t)> lhs;
  std::function<bool(size_t)> rhs;
};

// Callable object to check if `op` adheres to a given interface.
struct VerifyInterface {
  VerifyInterface() = delete;
  explicit VerifyInterface(std::function<LogicalResult(Operation *op)> fun)
      : fun(fun){};

  bool operator()(Operation *op) {
    if (succeeded(fun(op)))
      return true;
    return false;
  }

  std::function<LogicalResult(Operation *op)> fun;
};

// Work-around for template specialization.
struct WithSingleOpImpl {
  WithSingleOpImpl() = default;

  bool withSingleOpImpl(StringRef, Region *, Operation *,
                        SmallVectorImpl<Value> *);
};

// Callable object to check the `op` region for a single scalar operation OpTy.
template <typename OpTy> struct WithSingleOp {
  WithSingleOp() = delete;
  WithSingleOp(SmallVectorImpl<Value> *captures) : captures(captures){};

  bool operator()(Region *region, Operation *op) {
    return WithSingleOpImpl().withSingleOpImpl(OpTy::getOperationName(), region,
                                               op, captures);
  }

private:
  SmallVectorImpl<Value> *captures;
};

class StructuredOpMatcher {
  using PredicateFn = std::function<bool(linalg::LinalgOp)>;

public:
  StructuredOpMatcher() = default;

  StructuredOpMatcher(PredicateFn &&firstPredicate) {
    predicates.push_back(std::move(firstPredicate));
  }

  template <typename OpTy> static StructuredOpMatcher make() {
    return StructuredOpMatcher(
        [](linalg::LinalgOp op) { return isa<OpTy>(op.getOperation()); });
  }

  // Match given `op` using stored predicates.
  bool match(Operation *op);

  // Predicates on operation.
  StructuredOpMatcher &operation(std::function<bool(Operation *)>);

  // Predicate on OpOperands.
  StructuredOpMatcher &input(MatchSelector range,
                             std::function<bool(OpOperand *, Operation *)>);

  // Predicates on OpOperands.
  StructuredOpMatcher &output(MatchSelector range,
                              std::function<bool(OpOperand *, Operation *)>);

  // Predicates on Iterators.
  StructuredOpMatcher &dim(MatchSelector range,
                           SmallVector<mlir::utils::IteratorType> kinds);
  StructuredOpMatcher &dim(MatchSelector range, mlir::utils::IteratorType kind);

  // Predicates on region.
  StructuredOpMatcher &region(MatchSelector range,
                              std::function<bool(Region *, Operation *op)>);

private:
  llvm::SmallVector<PredicateFn> predicates;
};

} // namespace structured_match
} // namespace tpp
} // namespace mlir

#endif // TPP_STRUCTUREDOPMATCHERS_H
