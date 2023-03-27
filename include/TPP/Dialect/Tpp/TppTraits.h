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

} // namespace tpp
} // namespace OpTrait
} // namespace mlir

#endif // TPP_DIALECT_TPP_TRAITS_H
