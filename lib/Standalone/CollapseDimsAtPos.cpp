#include "Standalone/Transforms.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "linalg-drop-unit-dims"

using namespace mlir;

FailureOr<linalg::GenericOp>
CollapseDimsAtPosForOperand(RewriterBase &rewriter, linalg::LinalgOp linalgOp,
                            OpOperand *operand, ArrayRef<int64_t> pos) {
  return failure();
}
