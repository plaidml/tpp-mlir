//===- TilingUtils.h - -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_IR_TILINGUTILS_H
#define TPP_IR_TILINGUTILS_H

#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"

using namespace mlir;
using namespace mlir::affine;

namespace mlir {
namespace tpp {
class FlatRelation : public FlatAffineValueConstraints {
  using FlatAffineValueConstraints::FlatAffineValueConstraints;

public:
  void removeLocalVars() { return removeRedundantLocalVars(); }

  void tightenInequalities() { return gcdTightenInequalities(); }

  AffineMap getAsNormalizedConstraintsMap(MLIRContext *context,
                                          int64_t projectedDimension);
};

int64_t getCumulativeTileSize(SmallVector<SmallVector<unsigned>> tilingVectors,
                              size_t k);

void getInverseOfAffineMap(SmallVector<unsigned> tilingVectors,
                           AffineValueMap map,
                           SmallVector<AffineMap> &composedMap);
} // namespace tpp
} // namespace mlir

#endif // TPP_IR_TILINGUTILS_H
