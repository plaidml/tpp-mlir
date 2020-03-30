#ifndef INCLUDE_STANDALONE_STANDALONEDIALECT_H
#define INCLUDE_STANDALONE_STANDALONEDIALECT_H

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace standalone {

class StandaloneDialect : public Dialect {
public:
  explicit StandaloneDialect(MLIRContext *context);

  static StringRef getDialectNamespace() { return "standalone"; }
};

} // namespace standalone
} // namespace mlir

#endif // INCLUDE_STANDALONE_STANDALONEDIALECT_H
