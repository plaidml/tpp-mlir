//===- VNNIDialect.h - VNNI dialect -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef VNNI_DIALECT_H_
#define VNNI_DIALECT_H_

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace vnni {

class VNNIDialect : public Dialect {
public:
  explicit VNNIDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "vnni"; }
};

} // namespace vnni
} // namespace mlir

#endif // VNNI_DIALECT_H_
