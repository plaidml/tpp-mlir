#ifndef STANDALONE_STANDALONEDIALECT_H
#define STANDALONE_STANDALONEDIALECT_H

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

#endif // STANDALONE_STANDALONEDIALECT_H
