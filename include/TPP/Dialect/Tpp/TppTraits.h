#ifndef TPP_DIALECT_TPP_TRAITS_H
#define TPP_DIALECT_TPP_TRAITS_H

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace OpTrait {
namespace tpp {

LogicalResult verifyBroadcastableShapeImpl(Operation *op);
LogicalResult verifyUnitStrideInnerLoopImpl(Operation *op);

template <typename ConcreteType>
struct BroadcastableShape
    : public OpTrait::TraitBase<ConcreteType, BroadcastableShape> {
  static LogicalResult verifyTrait(Operation *op) {
    return verifyBroadcastableShapeImpl(op);
  }
};

template <typename ConcreteType>
struct UnitStrideInnerLoop
    : public OpTrait::TraitBase<ConcreteType, UnitStrideInnerLoop> {
  static LogicalResult verifyTrait(Operation *op) {
    return verifyUnitStrideInnerLoopImpl(op);
  }
};

} // namespace tpp
} // namespace OpTrait
} // namespace mlir

#endif // TPP_DIALECT_TPP_TRAITS_H
