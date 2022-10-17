// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CHECK_DIALECT_H_
#define CHECK_DIALECT_H_

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace check {

class CheckDialect : public Dialect {
public:
  explicit CheckDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "check"; }
};

} // namespace check
} // namespace mlir

#endif // CHECK_DIALECT_H_
