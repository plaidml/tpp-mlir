#include "Standalone/StandaloneOps.h"
#include "Standalone/StandaloneDialect.h"

using namespace mlir;

namespace mlir {
namespace standalone {
#define GET_OP_CLASSES
#include "Standalone/StandaloneOps.cpp.inc"
} // namespace standalone
} // namespace mlir
