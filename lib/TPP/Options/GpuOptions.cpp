//===- GpuOptions.cpp --------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Options/GpuOptions.h"

#include "mlir/Support/LLVM.h"

using namespace mlir;

namespace mlir {
namespace tpp {
namespace opt {

// Select target GPU backend for the pipeline.
llvm::cl::opt<std::string>
    defGpuBackend("gpu", llvm::cl::desc("Target GPU backend for lowering"),
                  llvm::cl::value_desc("cuda,intel"), llvm::cl::init(""));

// Kernel buffers - arguments and return values - are expected to be allocated
// on GPU.
llvm::cl::opt<bool>
    defGpuArgs("gpu-args",
               llvm::cl::desc("Kernel buffers are allocated on GPU"),
               llvm::cl::init(true));

llvm::cl::list<int64_t>
    gpuBlockTile("gpu-block-tile", llvm::cl::desc("GPU block tile size"),
                 llvm::cl::list_init<int64_t>(SmallVector<int64_t>{128, 128}),
                 llvm::cl::CommaSeparated);

llvm::cl::list<int64_t>
    gpuThreadTile("gpu-thread-tile", llvm::cl::desc("GPU thread tile size"),
                  llvm::cl::list_init<int64_t>(SmallVector<int64_t>{32, 32}),
                  llvm::cl::CommaSeparated);

llvm::cl::opt<int64_t> kTile("k-tile", llvm::cl::desc("GEMM K dim tiling size"),
                             llvm::cl::init(32));

llvm::cl::opt<int64_t> stages("stages",
                              llvm::cl::desc("GEMM coop prefetch stages"),
                              llvm::cl::init(1));

// DPAS size defaults to PVC.
llvm::cl::list<int64_t>
    gpuDpasTile("dpas-tile", llvm::cl::desc("DPAS register block sizes MxNxK"),
                llvm::cl::list_init<int64_t>(SmallVector<int64_t>{8, 16, 16}),
                llvm::cl::CommaSeparated);

// Control GPU vectorization.
llvm::cl::opt<bool> gpuVector("gpu-vector",
                              llvm::cl::desc("Vectorize GPU kernel"),
                              llvm::cl::init(false));

} // namespace opt
} // namespace tpp
} // namespace mlir
