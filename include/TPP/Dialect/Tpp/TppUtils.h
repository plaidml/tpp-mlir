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

// Returns true if the linalg generic is a tpp.matmul.
// 1. Buffer semantics.
// 2. One output and two inputs.
// 3. A single region with a an add and a mul operation
// 4. Loops are: [parallel, parallel, reduction]
// 5. Access pattern is [i, j] -> [i, k] [k, j]
bool isTppMatmul(linalg::GenericOp linalgOp);

// Returns true if the linalg generic is a tpp.add.
// 1. Buffer semantics.
// 2. One output and one input.
// 3. A single region with an add operation.
// 4. All loops are parallel and the number of loops is less than or equal to 2.
bool isTppAdd(linalg::GenericOp linalgOp);

// Returns true if the linalg.generic is a tpp.identity
// 1. Buffer semantics.
// 2. One output and one input.
// 3. A single region with a yield operation.
// 4. All loops are parallel and the number of loops is less than or equal to 2.
bool isTppIdentity(linalg::GenericOp linalgOp);

// Returns true if the linalg.generic is a tpp.relu:
// 1. Buffer semantics.
// 2. One output and zero input.
// 3. A single region with a maxf operation.
// 4. All loops are parallel and the number of loops is less than or equal to 2.
bool isTppRelu(linalg::GenericOp linalgOp);

// Returns true if the linalg generic can be mapped to a tpp.identity.
bool canMapToTppIdentity(linalg::GenericOp linalgOp);

// Returns true if the linalg generic can be mapped to a tpp.relu.
bool canMapToTppRelu(linalg::GenericOp linalgOp);

// Returns true if the linalg generic can be mapped to a tpp.add.
bool canMapToTppAdd(linalg::GenericOp linalgOp);

} // namespace utils
} // namespace tpp
} // namespace mlir

#endif // TPP_UTILS_H
