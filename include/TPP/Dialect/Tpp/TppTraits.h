#ifndef TPP_DIALECT_TPP_TRAITS_H
#define TPP_DIALECT_TPP_TRAITS_H

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace OpTrait {
namespace tpp {

LogicalResult verifyBroadcastableShape(Operation *op,
                                       bool emitDiagnostic = true);
LogicalResult verifyUnitStrideInnerLoop(Operation *op,
                                        bool emitDiagnostic = true);

LogicalResult checkBroadcastableShape(Operation *op);
LogicalResult checkUnitStrideInnerLoop(Operation *op);
LogicalResult verifyArity(Operation *op, unsigned numInput);

template <typename ConcreteType>
struct BroadcastableShape
    : public OpTrait::TraitBase<ConcreteType, BroadcastableShape> {
  static LogicalResult verifyTrait(Operation *op) {
    return verifyBroadcastableShape(op);
  }
};

template <typename ConcreteType>
struct UnitStrideInnerLoop
    : public OpTrait::TraitBase<ConcreteType, UnitStrideInnerLoop> {
  static LogicalResult verifyTrait(Operation *op) {
    return verifyUnitStrideInnerLoop(op);
  }
};

template <typename ConcreteType>
struct UnaryOp : public OpTrait::TraitBase<ConcreteType, UnaryOp> {
  static LogicalResult verifyTrait(Operation *op) { return verifyArity(op, 1); }
};

template <typename ConcreteType>
struct BinaryOp : public OpTrait::TraitBase<ConcreteType, BinaryOp> {
  static LogicalResult verifyTrait(Operation *op) { return verifyArity(op, 2); }
};

template <typename ConcreteType>
struct TernaryOp : public OpTrait::TraitBase<ConcreteType, TernaryOp> {
  static LogicalResult verifyTrait(Operation *op) { return verifyArity(op, 3); }
};

template <typename ConcreteType>
struct QuaternaryOp : public OpTrait::TraitBase<ConcreteType, QuaternaryOp> {
  static LogicalResult verifyTrait(Operation *op) { return verifyArity(op, 4); }
};

} // namespace tpp
} // namespace OpTrait
} // namespace mlir

#endif // TPP_DIALECT_TPP_TRAITS_H
