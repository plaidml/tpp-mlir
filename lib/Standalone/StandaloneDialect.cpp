#include "Standalone/StandaloneDialect.h"
#include "Standalone/StandaloneOps.h"

using namespace mlir;
using namespace mlir::standalone;

//===----------------------------------------------------------------------===//
// Standalone dialect.
//===----------------------------------------------------------------------===//

StandaloneDialect::StandaloneDialect(mlir::MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {}
