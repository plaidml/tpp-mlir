//===- matmul-cuda.cpp -------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Bench.h"
#include "Config.h"
#include "CudaTensor.h"
#include "Tensor.h"

#include <iomanip>
#include <iostream>

#include <cublas_v2.h>

struct MatmulKernelCUBLAS : public KernelInterface<CudaTensor<float>> {
  MatmulKernelCUBLAS() {
    isInit = cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS;
    if (!isInit)
      std::cerr << "CUBLAS initialization failed!\n";

    cublasSetStream(handle, stream);
  }

  ~MatmulKernelCUBLAS() { cublasDestroy(handle); }

  void runRef(std::vector<CudaTensor<float>> &args) override {
    assert(args.size() == 3 && "wrong rank for MLP");

    if (!isInit)
      return;

    auto &a = args[0];
    auto &b = args[1];
    auto &o = args[2];

    // MATMUL O += A x B
    int m = o.tensor.getDim(0);
    int n = o.tensor.getDim(1);
    int k = a.tensor.getDim(1);

    auto transa = CUBLAS_OP_N;
    auto transb = CUBLAS_OP_N;
    float alpha = 1.0f;
    float beta = 1.0f;

    // See:
    // https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication
    // Swap A and B, and col-major change m with n.
    float *A = b.gpuData;
    float *B = a.gpuData;
    float *C = o.gpuData;
    int lda = n;
    int ldb = k;
    int ldc = n;

    // printf("m=%d, n=%d, k=%d\n", n, m, k);
    // printf("lda=%d, ldb=%d, ldc=%d\n", lda, ldb, ldc);

    cublasStatus_t gemmStatus = cublasSgemm(
        handle, transa, transb, n, m, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
    if (gemmStatus != CUBLAS_STATUS_SUCCESS) {
      cudaError_t cudaStatus = cudaGetLastError();
      std::cerr << "cublasSgemm error : cublas code=" << gemmStatus
                << " cuda code=" << cudaStatus << " - "
                << cudaGetErrorString(cudaStatus) << "\n";
    }

    cudaError_t syncStatus = cudaDeviceSynchronize();
    if (syncStatus != cudaSuccess) {
      std::cerr << "cudaDeviceSynchronize error : cuda code=" << syncStatus
                << " - " << cudaGetErrorString(syncStatus);
    }
  }

  cublasHandle_t handle;
  cudaStream_t stream = 0;
  bool isInit = false;
};

int main(int argc, char *argv[]) {
  // Assume success until proven wrong
  int returnValue = 0;

  // These need to be from the command line
  unsigned m = 0;
  unsigned n = 0;
  unsigned k = 0;

  // Cmd-line args
  BenchConfig config(argc, argv);
  if (config.dims.size() == 3) {
    m = config.dims[0];
    n = config.dims[1];
    k = config.dims[2];
  } else {
    std::cerr << "--input argument required to be 3D, use --help for options\n";
    return 1;
  }

  if (config.verbose) {
    std::cerr << "Kernel version: "
              << "ref" << std::endl;
    std::cerr << "[ " << m << ", " << n << " ] = "
              << "[ " << m << ", " << k << " ] * "
              << "[ " << k << ", " << n << " ] X " << config.iter << std::endl;
  }

  CudaTensor<float> gpuA(std::move(ConstantTensor<float>{m, k}));
  CudaTensor<float> gpuB(std::move(ConstantTensor<float>{k, n}));
  CudaTensor<float> gpuC(std::move(EmptyTensor<float>{m, n}));

  if (!gpuA.initGpu() || !gpuB.initGpu() || !gpuC.initGpu())
    return 1;

  double gflops = static_cast<double>(2 * n * m * k) / 1e9;
  auto bench =
      Benchmark<MatmulKernelCUBLAS, CudaTensor<float>>(config.iter, gflops);
  bench.setArg({gpuA, gpuB, gpuC});

  // Warmup (TODO: Check output)
  bench.warmup();

  // Run the reference benchmark
  bench.run();

  double mean = bench.getMean();
  double stdev = bench.getStdev();
  std::string unit = "ms";
  if (gflops)
    unit = "gflops";

  std::cout << std::fixed << std::setw(9) << std::setprecision(3) << mean
            << " +- " << std::setw(9) << stdev << " " << unit << std::endl;

  return returnValue;
}