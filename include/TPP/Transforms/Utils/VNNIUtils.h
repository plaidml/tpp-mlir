//===- VNNIUtils.h -----------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_TRANSFORMS_UTILS_VNNIUTILS_H
#define TPP_TRANSFORMS_UTILS_VNNIUTILS_H

#include "mlir/Support/LogicalResult.h"
#include <cstdint>
#include <optional>

namespace mlir {
class Type;
class MemRefType;
class OpOperand;
class AffineDimExpr;
class AffineMap;
class VectorType;
class Operation;

namespace linalg {
class LinalgOp;
} // namespace linalg

namespace vnni {
namespace utils {

enum class VnniOperandRank {
  TRANSPOSE = 3,
  GEMM = 3,
  BRGEMM_INS = 4,
  BRGEMM_OUTS = 3
};

// Return the VNNI blocking factor.
// Optionally, operation can be provided to give access to DLTI.
std::optional<int64_t> getVnniBlockingFactor(Type type,
                                             Operation *op = nullptr);

// Return true if the memref is in VNNI layout with rank `expectedRank`.
bool isInVnniLayout(VnniOperandRank expectedRank, MemRefType memref);

// Return true if the vector is in VNNI layout with rank `expectedRank`.
bool isInVnniLayout(VnniOperandRank expectedRank, VectorType vector);

bool isInVnniLayout(int64_t expectedRank, VectorType vector);

// Return true if the operation is in VNNI layout.
// Optionally, the check can be constrained to a specific VNNI blocking factor.
bool isInVnniLayout(linalg::LinalgOp linalgOp,
                    std::optional<int64_t> blockingFactor = std::nullopt);

} // namespace utils
} // namespace vnni
} // namespace mlir

#endif
