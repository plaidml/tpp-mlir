//===- GpuOptions.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Collection of CLI flags controlling GPU pipeline.
//
//===----------------------------------------------------------------------===//

#ifndef TPP_OPTIONS_GPUOPTIONS_H
#define TPP_OPTIONS_GPUOPTIONS_H

#include "llvm/Support/CommandLine.h"

#include <string>

namespace mlir {
namespace tpp {
namespace opt {

// Select target GPU backend for the pipeline.
extern llvm::cl::opt<std::string> defGpuBackend;

// Kernel buffers - arguments and return values - are expected to be allocated
// on GPU.
extern llvm::cl::opt<bool> defGpuArgs;

extern llvm::cl::list<int64_t> gpuBlockTile;

extern llvm::cl::list<int64_t> gpuThreadTile;

extern llvm::cl::opt<int64_t> kTile;

extern llvm::cl::opt<int64_t> stages;

// DPAS size defaults to PVC.
extern llvm::cl::list<int64_t> gpuDpasTile;

// Control GPU vectorization.
extern llvm::cl::opt<bool> gpuVector;

} // namespace opt
} // namespace tpp
} // namespace mlir

#endif // TPP_OPTIONS_GPUOPTIONS_H
