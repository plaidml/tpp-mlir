//===- DLTIUtils.h -----------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_TRANSFORMS_UTILS_DLTIUTILS_H
#define TPP_TRANSFORMS_UTILS_DLTIUTILS_H

#include "mlir/Dialect/DLTI/DLTI.h"

namespace llvm {
class StringRef;
} // namespace llvm

namespace mlir {
namespace dlti {
namespace utils {

// Perform a DLTI-query using string keys.
FailureOr<Attribute> query(Operation *op, ArrayRef<StringRef> keys,
                           bool emitError = false);

} // namespace utils
} // namespace dlti
} // namespace mlir

#endif // TPP_TRANSFORMS_UTILS_DLTIUTILS_H
