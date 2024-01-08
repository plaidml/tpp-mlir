//===- PassUtils.h - Helpers for pass creation ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities to help build MLIR passes.
//
//===----------------------------------------------------------------------===//

#ifndef TPP_PASSUTILS_H
#define TPP_PASSUTILS_H

#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace tpp {

// Helper base class for passes that call and manage combination of other
// existing passes.
template <typename OpT> class UtilityPassBase {
public:
  UtilityPassBase()
      : pm(OpT::getOperationName(), mlir::OpPassManager::Nesting::Implicit){};
  virtual ~UtilityPassBase() = default;

protected:
  OpPassManager pm;

  // Create the pass processing pipeline.
  virtual void constructPipeline() = 0;
};

} // namespace tpp
} // namespace mlir

#endif // TPP_PASSUTILS_H
