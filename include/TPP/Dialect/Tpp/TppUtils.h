//===- TPPUtils.h - ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_UTILS_H
#define TPP_UTILS_H

#include <string>

namespace mlir {

namespace linalg {
class LinalgOp;
class GenericOp;
} // end namespace linalg

namespace tpp {
namespace utils {

// Return true if all the operands of the linalg operation have static
// dimensions.
bool hasStaticShape(linalg::LinalgOp linalgOp);

// Return true if the linalg operation has been marked by the tpp detection pass
// and the operation can be mapped to a tpp operation.
bool hasTppMark(linalg::LinalgOp linalgOp);

// Return true if the linalg operation is marked with 'target'.
bool isMarkedWithTpp(linalg::LinalgOp linalgOp, const std::string &target);

// Return true if the linalg operation has a Matmul region.
bool hasMatmulBody(linalg::LinalgOp linalgOp);

// Return true if the linalg operation has copy semantics.
bool hasCopySemantics(linalg::LinalgOp linalgOp);

// Return true if the linalg generic region contains a maxf(x, 0) operation.
bool hasMaxfZeroOp(linalg::LinalgOp linalgOp);

// Return true if the linalg generic has 1 input 1 output.
bool hasOneInputOneOutput(linalg::GenericOp);

// Return true if the linalg generic can be mapped to a tpp.matmul (aka Gemm)
// operation.
bool isTPPGemm(linalg::GenericOp genericOp);

// Return true if the linalg generic can be mapped to a tpp.relu.
bool isTPPRelu(linalg::GenericOp genericOp);

// Return true if the linalg generic can be mapped to a tpp.add.
bool isTPPAdd(linalg::GenericOp genericOp);

// Return true if the linalg generic can be mapped to a tpp.identity.
bool isTPPIdentity(linalg::GenericOp genericOp);

} // namespace utils
} // namespace tpp
} // namespace mlir

#endif // TPP_UTILS_H
