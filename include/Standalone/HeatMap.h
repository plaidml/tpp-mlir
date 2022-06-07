//===- HeatMap.h - ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TPP_HEATMAP_H
#define MLIR_TPP_HEATMAP_H

#include "llvm/ADT/ArrayRef.h"

namespace mlir {
namespace tpp {
namespace x86 {

struct KernelCost {
  double throughput;
  unsigned startupCost;
};

// Return the cost of 'tile' by heatmap lookup.
KernelCost lookupHeatMap(llvm::ArrayRef<int64_t> tile);

} // end namespace mlir
} // end namespace tpp
} // end namespace x86

#endif // MLIR_TPP_HEATMAP_H
