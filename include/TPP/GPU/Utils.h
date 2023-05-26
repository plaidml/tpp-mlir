//===- Utils.h - GPU-related helpers --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_GPU_UTILS_H
#define TPP_GPU_UTILS_H

namespace mlir {
namespace tpp {

void initializeGpuTargets();

} // namespace tpp
} // namespace mlir

#endif // TPP_GPU_UTILS_H
