//===- TPPUtils.h - ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_DIALECT_TPP_TPPUTILS_H
#define TPP_DIALECT_TPP_TPPUTILS_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include <string>

namespace mlir {
class PatternRewriter;

namespace linalg {
class LinalgOp;
class GenericOp;
class YieldOp;
} // end namespace linalg

namespace tpp {
class FusedBrgemmOp;

namespace utils {

// Splits and replaces fused op with its individual components.
// Temporary workaround for:
// https://github.com/libxsmm/libxsmm/issues/766
// TODO: Move into tpp-to-loops as a private helper.
LogicalResult splitAndReplaceFusedOp(tpp::FusedBrgemmOp fusedBrgemmOp,
                                     PatternRewriter &rewriter);

} // namespace utils
} // namespace tpp
} // namespace mlir

#endif // TPP_DIALECT_TPP_TPPUTILS_H
