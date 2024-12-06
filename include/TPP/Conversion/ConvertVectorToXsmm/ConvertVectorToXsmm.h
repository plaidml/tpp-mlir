#include "mlir/Pass/Pass.h"
using namespace std;

namespace mlir {
namespace tpp {
std::unique_ptr<mlir::Pass> createConvertVectorToXsmm();
}
} // namespace mlir
