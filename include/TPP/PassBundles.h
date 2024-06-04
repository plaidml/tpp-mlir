//===- PassBundles.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_PASSBUNDLES_H
#define TPP_PASSBUNDLES_H

#include "TPP/Passes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace tpp {
#define GEN_PASS_DECL
#include "TPP/PassBundles.h.inc"

#define GEN_PASS_REGISTRATION
#include "TPP/PassBundles.h.inc"
} // namespace tpp
} // namespace mlir

#endif // TPP_PASSBUNDLES_H
