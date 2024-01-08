//===- CheckDialect.h - Check dialect ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_DIALECT_CHECK_CHECKDIALECT_H
#define TPP_DIALECT_CHECK_CHECKDIALECT_H

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

#endif // TPP_DIALECT_CHECK_CHECKDIALECT_H
