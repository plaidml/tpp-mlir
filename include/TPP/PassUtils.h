//===- Pass Utils - Helper for pass creation  -----------------------------===//
// Utilities to help build MLIR passes
//
//===----------------------------------------------------------------------===//

#ifndef TPP_PASS_UTILS_H
#define TPP_PASS_UTILS_H

#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace tpp {

// Helper base class for passes that call and manage combination of other
// existing passes.
template <typename OpT> class UtilityPassBase {
public:
  UtilityPassBase()
      : pm(OpT::getOperationName(), mlir::OpPassManager::Nesting::Implicit){};
  virtual ~UtilityPassBase() = default;

protected:
  OpPassManager pm;

  // Create the pass processing pipeline.
  virtual void constructPipeline() = 0;
};

} // namespace tpp
} // namespace mlir

#endif // TPP_PASS_UTILS_H
