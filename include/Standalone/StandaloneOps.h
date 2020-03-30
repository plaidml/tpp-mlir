#ifndef STANDALONE_STANDALONEOPS_H
#define STANDALONE_STANDALONEOPS_H

namespace mlir {
namespace standalone {

#define GET_OP_CLASSES
#include "Standalone/StandaloneOps.h.inc"

} // namespace standalone
} // namespace mlir

#endif // STANDALONE_STANDALONEOPS_H
