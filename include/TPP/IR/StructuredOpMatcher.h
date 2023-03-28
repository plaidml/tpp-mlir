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

struct AllDimsBut {
  AllDimsBut() = delete;
  explicit AllDimsBut(std::initializer_list<size_t> range) {
    llvm::append_range(exceptions, range);
  }
  explicit AllDimsBut(RangeDims range) {
    for (auto i = range.getLowerBound(); i < range.getUpperBound(); i++)
      exceptions.push_back(i);
  }
  ArrayRef<size_t> getExceptions() const { return exceptions; }

private:
  SmallVector<size_t> exceptions;
};

struct AllOperands {};
struct Operand {
  Operand() = delete;
  Operand(size_t idx) : idx(idx){};

  const size_t idx;
};

struct IsProjectedPermutation {
  IsProjectedPermutation() = default;
  bool operator()(AffineMap map) const {
    return map.isProjectedPermutation(/*allowZeroInResults=*/true);
  }
};

struct IsIdentity {
  IsIdentity() = default;
  bool operator()(AffineMap map) const { return map.isIdentity(); }
};

struct HasStaticShape {};

struct NumEqualsTo {
  NumEqualsTo() = delete;
  explicit NumEqualsTo(size_t value) : value(value){};
  const size_t value;

  bool operator()(size_t value) { return value == this->value; }
};

struct LessThanOrEqualTo {
  LessThanOrEqualTo() = delete;
  explicit LessThanOrEqualTo(size_t value) : value(value){};
  const size_t value;

  bool operator()(size_t value) { return value <= this->value; }
};

struct GreaterThanOrEqualTo {
  GreaterThanOrEqualTo() = delete;
  explicit GreaterThanOrEqualTo(size_t value) : value(value){};
  const size_t value;

  bool operator()(size_t value) const { return value >= this->value; };
};

struct HasAffineMapEqualsTo {
  HasAffineMapEqualsTo() = delete;
  explicit HasAffineMapEqualsTo(AffineMap map) : map(map){};
  AffineMap map;

  bool operator()(AffineMap map) const { return map == this->map; }
};

// OR or AND between predicates (functors).
struct BinaryPredicate {
public:
  enum BinaryPredicateKind { _OR, _AND };

private:
  const BinaryPredicateKind binaryPredicateKind;

public:
  std::function<bool(size_t)> predicateOnLhs;
  std::function<bool(size_t)> predicateOnRhs;

public:
  BinaryPredicateKind getKind() const { return binaryPredicateKind; }
  BinaryPredicate() = delete;
  BinaryPredicate(BinaryPredicateKind binaryPredicateKind,
                  std::function<bool(size_t)> lhs,
                  std::function<bool(size_t)> rhs)
      : binaryPredicateKind(binaryPredicateKind),
        predicateOnLhs(std::move(lhs)), predicateOnRhs(std::move(rhs)) {}
};

struct _OR : public BinaryPredicate {
  _OR(std::function<bool(size_t)> lhs, std::function<bool(size_t)> rhs)
      : BinaryPredicate(BinaryPredicateKind::_OR, lhs, rhs) {}

  static bool classof(const BinaryPredicate *binaryPredicate) {
    return binaryPredicate->getKind() == BinaryPredicateKind::_OR;
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
  StructuredOpMatcher &hasBufferSemantics();
  StructuredOpMatcher &hasTensorSemantics();
  StructuredOpMatcher &inputs(std::function<bool(size_t)>);
  StructuredOpMatcher &outputs(std::function<bool(size_t)>);
  StructuredOpMatcher &inputs(BinaryPredicate);
  StructuredOpMatcher &
      verifyInterface(std::function<LogicalResult(Operation *op)>);

  // Predicate on OpOperands.
  StructuredOpMatcher &input(AllOperands tag, HasStaticShape);
  StructuredOpMatcher &input(AllOperands tag, std::function<bool(AffineMap)>);
  StructuredOpMatcher &input(Operand operand,
                             std::function<bool(AffineMap map)>);

  // Predicates on OpOperands.
  StructuredOpMatcher &output(AllOperands tag, HasStaticShape);
  StructuredOpMatcher &output(AllOperands tag, std::function<bool(AffineMap)>);
  StructuredOpMatcher &output(Operand operand,
                              std::function<bool(AffineMap map)>);

  // Predicates on Iterators.
  StructuredOpMatcher &dim(std::function<bool(size_t)>);
  StructuredOpMatcher &dim(RangeDims range,
                           SmallVector<mlir::utils::IteratorType> kinds);
  StructuredOpMatcher &dim(RangeDims range, mlir::utils::IteratorType kind);
  StructuredOpMatcher &dim(AllDimsBut, mlir::utils::IteratorType kind);

  // Predicates on region.
  template <typename OpTy>
  StructuredOpMatcher &
  hasRegionWithSingleOp(SmallVectorImpl<Value> *capturedOperands) {
    return hasRegionWithSingleOpImpl(OpTy::getOperationName(),
                                     capturedOperands);
  }
  StructuredOpMatcher &
  hasRegion(std::function<bool(Operation *op,
                               SmallVectorImpl<Value> *capturedOperands)>,
            SmallVectorImpl<Value> *capturedOperands);

private:
  llvm::SmallVector<PredicateFn> predicates;

  StructuredOpMatcher &
  hasRegionWithSingleOpImpl(StringRef operationName,
                            SmallVectorImpl<Value> *capturedOperands);
};

} // namespace structured_match
} // namespace tpp
} // namespace mlir

#endif // TPP_STRUCTUREDOPMATCHERS_H
