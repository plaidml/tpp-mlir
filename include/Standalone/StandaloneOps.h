#ifndef INCLUDE_STANDALONE_STANDALONEOPS_H
#define INCLUDE_STANDALONE_STANDALONEOPS_H

namespace mlir {
namespace standalone {

#define GET_OP_CLASSES
#include "Standalone/StandaloneOps.h.inc"

} // namespace standalone
} // namespace mlir

#endif // INCLUDE_STANDALONE_STANDALONEOPS_H
