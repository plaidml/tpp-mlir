//===- Utils.cpp -------------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/GPU/Utils.h"

#include "llvm/Support/TargetSelect.h"

using namespace mlir;

namespace mlir {
namespace tpp {

void initializeGpuTargets() {
#ifdef TPP_GPU_ENABLE
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();
#endif // TPP_GPU_ENABLE
}

} // namespace tpp
} // namespace mlir
