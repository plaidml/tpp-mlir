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

struct AllDims {};
struct RangeDims {
  RangeDims() = delete;
  explicit RangeDims(AllDims)
      : lowerBound(0), upperBound(std::numeric_limits<size_t>::max()) {}
  explicit RangeDims(size_t lowerBound, size_t upperBound)
      : lowerBound(lowerBound), upperBound(upperBound) {
    assert(upperBound >= lowerBound);
  }
  size_t getLowerBound() const { return lowerBound; };
  size_t getUpperBound() const { return upperBound; };

private:
  const size_t lowerBound;
  const size_t upperBound;
};

struct AllOperands {};
struct Operand {
  Operand() = delete;
  Operand(size_t idx) : idx(idx){};

  const size_t idx;
};

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

struct EqualsTo {
  EqualsTo() = delete;
  explicit EqualsTo(size_t value) : value(value){};
  const size_t value;

  bool operator()(size_t value) const { return value == this->value; }
};

struct LessThanOrEqualTo {
  LessThanOrEqualTo() = delete;
  explicit LessThanOrEqualTo(size_t value) : value(value){};
  const size_t value;

  bool operator()(size_t value) const { return value <= this->value; }
};

struct GreaterThanOrEqualTo {
  GreaterThanOrEqualTo() = delete;
  explicit GreaterThanOrEqualTo(size_t value) : value(value){};
  const size_t value;

  bool operator()(size_t value) const { return value >= this->value; };
};

struct MapEqualsTo {
  MapEqualsTo() = delete;
  explicit MapEqualsTo(AffineMap map) : map(map){};
  AffineMap map;

  bool operator()(AffineMap map) const { return map == this->map; }
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

  bool withSingleOpImpl(StringRef, Operation *, SmallVectorImpl<Value> *);
};

// Callable object to check the `op` region for a single scalar operation OpTy.
template <typename OpTy> struct WithSingleOp {
  WithSingleOp() = default;

  bool operator()(Operation *op, SmallVectorImpl<Value> *captures) {
    return WithSingleOpImpl().withSingleOpImpl(OpTy::getOperationName(), op,
                                               captures);
  }
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

  bool match(Operation *op);

  // Predicates on operation.
  StructuredOpMatcher &operation(std::function<bool(Operation *)>);

  // Predicate on OpOperands.
  StructuredOpMatcher &input(AllOperands tag,
                             std::function<bool(OpOperand *, Operation *)>);
  StructuredOpMatcher &input(Operand tag,
                             std::function<bool(OpOperand *, Operation *)>);

  // Predicates on OpOperands.
  StructuredOpMatcher &output(AllOperands tag,
                              std::function<bool(OpOperand *, Operation *)>);
  StructuredOpMatcher &output(Operand tag,
                              std::function<bool(OpOperand *, Operation *)>);

  // Predicates on Iterators.
  StructuredOpMatcher &dim(RangeDims range,
                           SmallVector<mlir::utils::IteratorType> kinds);
  StructuredOpMatcher &dim(RangeDims range, mlir::utils::IteratorType kind);

  // Predicates on region.
  // TODO: To be consistent the region API should look like:
  // std::function<bool(Region* region, Operation *operation)>
  // How to pass the captures to the functors?
  StructuredOpMatcher &
  region(std::function<bool(Operation *op,
                            SmallVectorImpl<Value> *capturedOperands)>,
         SmallVectorImpl<Value> *capturedOperands);

private:
  llvm::SmallVector<PredicateFn> predicates;
};

} // namespace structured_match
} // namespace tpp
} // namespace mlir

#endif // TPP_STRUCTUREDOPMATCHERS_H
