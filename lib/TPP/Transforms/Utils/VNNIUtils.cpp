//===- VNNIUtils.cpp ---------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Transforms/Utils/VNNIUtils.h"
#include "libxsmm.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Vector/IR/VectorAttributes.h.inc"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/SetOperations.h"
#include <iostream>

namespace mlir {
namespace vnni {
namespace utils {

std::optional<int64_t> getVnniBlockingFactor(Type type) {
  auto elementType = getElementTypeOrSelf(type);
  if (elementType.isBF16())
    return libxsmm_cpuid_dot_pack_factor(LIBXSMM_DATATYPE_BF16);
  return std::nullopt;
}

// Until we have a better way to express the VNNI layout (see: #563), it is up
// to the callee to specify the expected rank in the VNNI layout as the rank
// depends on the operations we are dealing with.
bool isInVnniLayout(VnniOperandRank expectedRank, MemRefType memref) {
  if (memref.getRank() != static_cast<int64_t>(expectedRank) ||
      !memref.getElementType().isBF16()) {
    return false;
  }
  return memref.getShape().back() == vnni::utils::getVnniBlockingFactor(memref);
}

bool isInVnniLayout(int64_t expectedRank, VectorType vector) {
  if (vector.getRank() != expectedRank || !vector.getElementType().isBF16()) {
    return false;
  }
  return vector.getShape().back() == vnni::utils::getVnniBlockingFactor(vector);
}

// Until we have a better way to express the VNNI layout (see: #563), it is up
// to the callee to specify the expected rank in the VNNI layout as the rank
// depends on the operations we are dealing with.
bool isInVnniLayout(VnniOperandRank expectedRank, VectorType vector) {
  return isInVnniLayout((int64_t)expectedRank, vector);
}

FailureOr<AffineDimExpr> isInVnniLayout(linalg::GenericOp linalgOp,
                                        AffineMap map, int64_t blockingFactor) {
  ArrayRef<AffineExpr> results = map.getResults();
  SmallVector<mlir::utils::IteratorType> iteratorTypes =
      linalgOp.getIteratorTypesArray();

  AffineExpr vnniDim = results.back();
  auto dimExpr = dyn_cast<AffineDimExpr>(vnniDim);
  if (!dimExpr || iteratorTypes[dimExpr.getPosition()] !=
                      mlir::utils::IteratorType::reduction) {
    return failure();
  }

  for (auto result : results) {
    auto blockeDim = dyn_cast<AffineBinaryOpExpr>(result);
    if (!blockeDim)
      continue;
    if (blockeDim.getKind() != AffineExprKind::FloorDiv)
      continue;
    auto lhsDim = dyn_cast<AffineDimExpr>(blockeDim.getLHS());
    auto rhsCst = dyn_cast<AffineConstantExpr>(blockeDim.getRHS());
    if (!lhsDim || !rhsCst)
      continue;
    if (iteratorTypes[lhsDim.getPosition()] !=
        mlir::utils::IteratorType::reduction)
      continue;
    if (rhsCst.getValue() != blockingFactor)
      continue;
    return lhsDim;
  }
  return failure();
}

static llvm::SmallDenseSet<int64_t>
findPermutationsIndexingOperand(AffineMap indexingMap,
                                ArrayRef<mlir::vector::IteratorType> iterators,
                                mlir::vector::IteratorType iter) {
  assert(iterators.size() == indexingMap.getNumDims());
  llvm::SmallDenseSet<int64_t> res;
  for (AffineExpr e : indexingMap.getResults()) {
    if (auto d = dyn_cast<AffineDimExpr>(e)) {
      if (iterators[d.getPosition()] == iter &&
          llvm::count_if(indexingMap.getResults(), [d](AffineExpr e) {
            return e.isFunctionOfDim(d.getPosition());
          }) == 1)
        res.insert(d.getPosition());
    }
  }
  return res;
}

FailureOr<AffineDimExpr> isInVnniLayout(mlir::vector::ContractionOp contractOp,
                                        int64_t blockingFactor) {
  AffineMap map = contractOp.getIndexingMapsArray()[1];
  auto arrayAttr = contractOp.getIteratorTypes();
  SmallVector<mlir::vector::IteratorType> iteratorTypes;
  for (auto attr : arrayAttr) {
    iteratorTypes.push_back(
        cast<mlir::vector::IteratorTypeAttr>(attr).getValue());
  }

  int inputZeroRank =
      dyn_cast<ShapedType>(contractOp.getOperand(0).getType()).getRank();
  int inputOneRank =
      dyn_cast<ShapedType>(contractOp.getOperand(1).getType()).getRank();
  bool isVnni =
      isInVnniLayout(inputZeroRank, dyn_cast<VectorType>(
                                        contractOp.getOperand(0).getType())) &&
      isInVnniLayout(inputOneRank,
                     dyn_cast<VectorType>(contractOp.getOperand(1).getType()));
  if (!isVnni)
    return failure();
  ArrayRef<AffineExpr> results = map.getResults();

  AffineExpr vnniDim = results.back();
  auto dimExpr = dyn_cast<AffineDimExpr>(vnniDim);
  if (!dimExpr || iteratorTypes[dimExpr.getPosition()] !=
                      mlir::vector::IteratorType::reduction) {
    return failure();
  }
  AffineExpr rhsCst;
  for (auto result : results) {
    rhsCst = result;
    if (!rhsCst)
      continue;
    if (iteratorTypes[dyn_cast<AffineDimExpr>(rhsCst).getPosition()] !=
        mlir::vector::IteratorType::reduction)
      continue;
  }

  llvm::SmallDenseSet<int64_t> a = findPermutationsIndexingOperand(
      contractOp.getIndexingMapsArray()[0], iteratorTypes,
      vector::IteratorType::reduction);
  llvm::SmallDenseSet<int64_t> b = findPermutationsIndexingOperand(
      contractOp.getIndexingMapsArray()[1], iteratorTypes,
      vector::IteratorType::reduction);
  llvm::set_union(a, b);
  if (a.size() < 2) {
    return failure();
  }
  return dyn_cast<AffineDimExpr>(rhsCst);
}

} // namespace utils
} // namespace vnni
} // namespace mlir
