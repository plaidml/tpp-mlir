//===- TPPUtils.h - ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_DIALECT_TPP_TPPUTILS_H
#define TPP_DIALECT_TPP_TPPUTILS_H

#include <string>

namespace mlir {

namespace linalg {
class LinalgOp;
class GenericOp;
} // end namespace linalg

namespace tpp {
namespace utils {

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

// Returns true if linalg generic region contains a maxf(x, 0) operation.
bool hasMaxfZeroOp(linalg::LinalgOp linalgOp);

// Returns true if the linalg operation can convert to a tpp.matmul.
bool isTppMatmul(linalg::LinalgOp linalgOp);

// Returns true if the linalg operation can convert to a tpp.add.
bool isTppAdd(linalg::GenericOp linalgOp);

// Returns true if the linalg.generic can convert to a tpp.identity.
bool isTppIdentity(linalg::GenericOp linalgOp);

// Returns true if the linalg.generic can convert to a tpp.relu.
bool isTppRelu(linalg::GenericOp linalgOp);

} // namespace utils
} // namespace tpp
} // namespace mlir

#endif // TPP_DIALECT_TPP_TPPUTILS_H
