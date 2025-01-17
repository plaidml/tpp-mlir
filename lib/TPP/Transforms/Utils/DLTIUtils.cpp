//===- DLTIUtils.cpp ---------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Transforms/Utils/DLTIUtils.h"

namespace mlir {
namespace dlti {
namespace utils {

FailureOr<Attribute> query(Operation *op, ArrayRef<StringRef> keys,
                           bool emitError) {
  if (!op)
    return failure();

  auto ctx = op->getContext();
  SmallVector<DataLayoutEntryKey> entryKeys;
  for (auto &key : keys) {
    auto entry = StringAttr::get(ctx, key);
    entryKeys.push_back(entry);
  }

  return dlti::query(op, entryKeys, emitError);
}

} // namespace utils
} // namespace dlti
} // namespace mlir
