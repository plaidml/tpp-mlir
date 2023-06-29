//===- CudaTensor.h - -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "Tensor.h"
#include "llvm/ADT/STLExtras.h"

#include <cassert>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>

namespace {
using ListArg = std::vector<unsigned>;
using InitArg = std::initializer_list<unsigned>;
} // namespace

// CudaTensor: represents a tensor in GPU memory using CUDA.
// A wrapper around existing tensor on host.
template <typename T> struct CudaTensor {
  CudaTensor() = delete;
  CudaTensor(Tensor<T> &&tensor) : tensor(std::move(tensor)) {}

  virtual ~CudaTensor() {
    if (gpuData)
      cudaFree(gpuData);
  }

  bool initGpu() {
    if (gpuData)
      cudaFree(gpuData);

    auto dataSize = tensor.getDataSize();
    auto allocErr = cudaMalloc(&gpuData, dataSize);
    if (allocErr != cudaSuccess) {
      std::cerr << "GPU allocation error\n";
      return false;
    }

    auto data = tensor.getData();
    auto cpyErr = cudaMemcpy(gpuData, data, dataSize, cudaMemcpyHostToDevice);
    if (cpyErr != cudaSuccess) {
      std::cerr << "GPU memcpy error\n";
      return false;
    }

    return true;
  }

  T *gpuData = nullptr;
  Tensor<T> tensor;
};
