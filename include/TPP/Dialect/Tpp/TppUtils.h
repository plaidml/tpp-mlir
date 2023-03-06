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

namespace linalg {
class LinalgOp;
class GenericOp;
class YieldOp;
} // end namespace linalg

namespace tpp {
namespace utils {

struct OperandInfo {
  SmallVector<Value> inputs;
  SmallVector<Value> outputs;
};

// Returns true if all the operands of the linalg operation have static
// dimensions.
bool hasStaticShape(linalg::LinalgOp linalgOp);

// Returns true if the linalg operation has been marked by the tpp detection
// pass and the operation can be mapped to a tpp operation.
bool hasTppMark(linalg::LinalgOp linalgOp);

// Returns true if the linalg operation is marked with 'target'.
bool isMarkedWithTpp(linalg::LinalgOp linalgOp, const std::string &target);

// Returns true if the linalg operation has a Matmul region.
bool hasMatmulBody(linalg::LinalgOp linalgOp);

// Returns true if the linalg operation has copy semantics.
bool hasCopySemantics(linalg::LinalgOp linalgOp);

// Returns true if the linalg operation can convert to a tpp.matmul.
bool isTppMatmul(linalg::LinalgOp linalgOp);

// Returns true if the linalg operation can convert to a tpp.add.
bool isTppAdd(linalg::GenericOp linalgOp);

// Returns true if the linalg.generic can convert to a tpp.identity.
bool isTppIdentity(linalg::GenericOp linalgOp);

// Returns true if the linalg.generic can convert to a tpp.relu.
bool isTppRelu(linalg::GenericOp linalgOp, OperandInfo &info);

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

// Returns true if `types` have the same shape and strides. For example: A:
// memref<56x32xf32, strided<[32, 1], offset: ?>> B: memref<56x32xf32>
// allOperandsHaveSameShape(A, B) return true. C: memref<1x32xf32>
// allOperandsHaveSameShape(A, C) return false.
bool allOperandsHaveSameShapeAndStrides(TypeRange types);

// Check if tpp.identity satisfies broadcasting rules.
// see: https://numpy.org/doc/stable/reference/ufuncs.html#broadcasting
// TODO: Make this function general enough for other tpp ops when we support
// broadcasting.
enum class MatchBroadcastRuleResult {
  Success = 0,
  OutputNotShapedType,
  WrongOutputRank,
  FailedToVerifyRules,
};
MatchBroadcastRuleResult verifyTppIdentityBroadcastingRules(Type input,
                                                            Type output);

} // namespace utils
} // namespace tpp
} // namespace mlir

#endif // TPP_DIALECT_TPP_TPPUTILS_H
