#ifndef TPP_DIALECT_TPP_TRAITS_H
#define TPP_DIALECT_TPP_TRAITS_H

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace OpTrait {
namespace tpp {

LogicalResult verifyTraitImpl(Operation *op);

template <typename ConcreteType>
struct BroadcastableShape
    : public OpTrait::TraitBase<ConcreteType, BroadcastableShape> {
  static LogicalResult verifyTrait(Operation *op) {
    return verifyTraitImpl(op);
  }
};

} // namespace tpp
} // namespace OpTrait
} // namespace mlir

#endif // TPP_DIALECT_TPP_TRAITS_H
