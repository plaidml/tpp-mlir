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
class TypeRange;
class Value;
class PatternRewriter;

namespace linalg {
class LinalgOp;
class GenericOp;
class YieldOp;
} // end namespace linalg

namespace tpp {
class FusedBrgemmOp;

namespace utils {

// Returns true if the linalg operation is marked with 'target'.
// FIXME: This should be unnecessary but it's still used by convolutions
bool isMarkedWithTpp(linalg::LinalgOp linalgOp, const std::string &target);

// Returns true if the linalg operation has copy semantics.
bool hasCopySemantics(linalg::LinalgOp linalgOp);

// Returns true if the linalg operation can convert to a tpp.add.
bool isTppAdd(linalg::GenericOp linalgOp,
              SmallVectorImpl<Value> *capturedOperands = nullptr);

// Returns true if the linalg.generic can convert to a tpp.identity.
bool isTppIdentity(linalg::GenericOp linalgOp,
                   SmallVectorImpl<Value> *capturedOperands = nullptr);

// Returns true if the linalg.generic can convert to a tpp.zero.
bool isTppZero(linalg::GenericOp linalgOp,
               SmallVectorImpl<Value> *capturedOperands = nullptr);

// Returns true if the linalg.generic can convert to a tpp.relu.
bool isTppRelu(linalg::GenericOp linalgOp,
               SmallVectorImpl<Value> *capturedOperands = nullptr);

// Returns true if the linalg.generic can convert to a tpp.add + tpp.relu.
bool isTppBiasRelu(linalg::GenericOp linalgOp,
                   SmallVectorImpl<Value> *capturedOperands = nullptr);

// Return true if the linalg.generic can convert to a tpp.brgemm in VNNI format.
bool isTppVnniOp(linalg::GenericOp linalgOp,
                 SmallVectorImpl<Value> *capturedOperands = nullptr);

// Returns true if: 1) the region has a single block. 2) The block has a single
// operation `OP`. 3) The operation result types are int or float.
template <typename OP> static bool hasOnlyOp(Region &region) {
  if (!region.hasOneBlock())
    return false;
  unsigned numberOfOpsInRegion = 2;
  if (std::is_same<OP, linalg::YieldOp>::value)
    numberOfOpsInRegion = 1;
  if (std::distance(region.front().begin(), region.front().end()) !=
      numberOfOpsInRegion)
    return false;
  for (Operation &op : region.front()) {
    if (!isa<OP, linalg::YieldOp>(op) ||
        llvm::any_of(op.getResultTypes(),
                     [](Type type) { return !type.isIntOrFloat(); }))
      return false;
  }
  return true;
}

// Returns true if the value is a constant float or integer.
bool isValConstZero(Value val);

// Returns true if the op defining `val` represents a zero filled tensor.
bool isZeroTensor(Value val);

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
