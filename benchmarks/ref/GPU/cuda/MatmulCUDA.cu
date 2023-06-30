//===- MatmulCUDA.cu --------------------------------------------*- CUDA-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <math.h>

#include <cuda_runtime.h>

__global__ void matmulKernel(float *A, float *B, float *C, int m, int n,
                             int k) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= m || col >= n)
    return;

  float sum = 0;
  for (int i = 0; i < k; ++i) {
    sum += A[row * k + i] * B[i * n + col];
  }

  C[row * n + col] += sum;
}

void matmulCUDA(float *A, float *B, float *C, int m, int n, int k) {
  // Max thread per block: 1024 = 32 * 32
  const int maxThreadDim = 32;
  dim3 numThreads(m > maxThreadDim ? maxThreadDim : m,
                  n > maxThreadDim ? maxThreadDim : n);
  dim3 numBlocks(std::ceil(m / numThreads.x), std::ceil(n / numThreads.y));

  matmulKernel<<<numBlocks, numThreads>>>(A, B, C, m, n, k);
}
