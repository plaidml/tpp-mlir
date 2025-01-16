//===- VNNIUtils.cpp ---------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Transforms/Utils/VNNIUtils.h"
#include "TPP/Transforms/Utils/DLTIUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"

#include "libxsmm.h"

namespace mlir {
namespace vnni {
namespace utils {

std::optional<int64_t> getVnniBlockingFactor(Type type, Operation *op) {
  int64_t blockingFactor = 0;

  auto elementType = getElementTypeOrSelf(type);
  if (elementType.isBF16()) {
    // Check if a VNNI factor hint is associated to the IR via DLTI.
    auto vnniValue = dlti::utils::query(op, {"CPU", "vnni"});
    if (succeeded(vnniValue)) {
      if (auto intAttr = llvm::dyn_cast<IntegerAttr>(*vnniValue))
        blockingFactor = intAttr.getInt();
    } else {
      blockingFactor = libxsmm_cpuid_dot_pack_factor(LIBXSMM_DATATYPE_BF16);
    }
  }

  if (blockingFactor != 0 && blockingFactor % 2 == 0)
    return blockingFactor;

  return std::nullopt;
}

bool isInVnniLayout(linalg::LinalgOp linalgOp,
                    std::optional<int64_t> blockingFactor) {
  // Narrow down type operations - VNNI only applies to contractions.
  if (!linalg::isaContractionOpInterface(linalgOp))
    return false;

  auto matA = linalgOp->getOperand(0);
  auto matB = linalgOp->getOperand(1);
  auto typeA = dyn_cast<ShapedType>(matA.getType());
  auto typeB = dyn_cast<ShapedType>(matB.getType());
  unsigned rankA = typeA.getRank();
  unsigned rankB = typeB.getRank();
  // VNNI format requires at least 1 parallel and 2 reduction dimensions.
  if (rankA < 3 || rankB < 3)
    return false;

  FailureOr<linalg::ContractionDimensions> dims =
      linalg::inferContractionDims(linalgOp);
  if (failed(dims))
    return false;

  // At least two reduction dimensions are expected:
  // one for the VNNI factor and one for the K dimension
  if (dims->k.size() < 2)
    return false;

  // Validate affine maps - VNNI computation should be defined by the two
  // innermost reduction iterators.
  // The input matrix dimensions layout must match the following:
  //   - matrix A - [...][K/vnniFactor][vnniFactor]
  //   - matrix B - [...][K/vnniFactor][N][vnniFactor]
  SmallVector<mlir::utils::IteratorType> iteratorTypes =
      linalgOp.getIteratorTypesArray();
  AffineMap mapA = linalgOp.getMatchingIndexingMap(&linalgOp->getOpOperand(0));
  AffineMap mapB = linalgOp.getMatchingIndexingMap(&linalgOp->getOpOperand(1));

  auto vnniDimA = dyn_cast<AffineDimExpr>(mapA.getResult(rankA - 1));
  auto vnniDimB = dyn_cast<AffineDimExpr>(mapB.getResult(rankB - 1));
  if (!vnniDimA || !vnniDimB || vnniDimA != vnniDimB ||
      iteratorTypes[vnniDimA.getPosition()] !=
          mlir::utils::IteratorType::reduction)
    return false;
  auto redDimA = dyn_cast<AffineDimExpr>(mapA.getResult(rankA - 2));
  auto redDimB = dyn_cast<AffineDimExpr>(mapB.getResult(rankB - 3));
  if (!redDimA || !redDimB || redDimA != redDimB ||
      iteratorTypes[redDimA.getPosition()] !=
          mlir::utils::IteratorType::reduction)
    return false;
  auto parallelDimB = dyn_cast<AffineDimExpr>(mapB.getResult(rankB - 2));
  if (!parallelDimB || iteratorTypes[parallelDimB.getPosition()] !=
                           mlir::utils::IteratorType::parallel)
    return false;

  // VNNI factor must be:
  //   - the innermost inputs' dimension
  //   - statically known
  //   - multiple of 2 or equal to the specified factor
  auto vnniDimSize = typeB.getShape().back();
  if (!(vnniDimSize != ShapedType::kDynamic &&
        typeA.getShape().back() == vnniDimSize &&
        (blockingFactor ? vnniDimSize == *blockingFactor
                        : vnniDimSize % 2 == 0)))
    return false;

  // The split reduction dimension size should also match.
  if (typeA.getShape().end()[-2] != typeB.getShape().end()[-3])
    return false;

  return true;
}

bool isInVnniLayout(VnniOperandRank expectedRank, ShapedType shape,
                    std::optional<int64_t> blockingFactor) {
  return isInVnniLayout(static_cast<int64_t>(expectedRank), shape,
                        blockingFactor);
}

bool isInVnniLayout(int64_t expectedRank, ShapedType shape,
                    std::optional<int64_t> blockingFactor) {
  if (shape.getRank() != expectedRank || !shape.getElementType().isBF16())
    return false;

  if (shape.getShape().back() % 2 != 0)
    return false;

  if (blockingFactor && shape.getShape().back() != *blockingFactor)
    return false;

  return true;
}

} // namespace utils
} // namespace vnni
} // namespace mlir
