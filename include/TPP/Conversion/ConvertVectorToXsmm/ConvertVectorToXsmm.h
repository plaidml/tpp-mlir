#include "mlir/Pass/Pass.h"
using namespace std;
using namespace mlir;

namespace mlir {
namespace tpp {
std::unique_ptr<mlir::Pass> createConvertVectorToXsmm();
}
} // namespace mlir
