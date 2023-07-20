//===- MatmulRef.cpp ---------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Bench.h"
#include "Config.h"
#include "CudaTensor.h"
#include "MatmulCUDA.h"
#include "Tensor.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/CommandLine.h"

#include <iomanip>
#include <iostream>
#include <optional>

#include <cublas_v2.h>

namespace {
llvm::cl::opt<std::string> kernelType{
    "kernel", llvm::cl::desc("Kernel type (cuda, cublas)"),
    llvm::cl::init("cublas")};

enum class KernelType {
  Cuda,
  Cublas,
};

KernelType parseKernelOption(llvm::StringRef opt) {
  auto type = llvm::StringSwitch<std::optional<KernelType>>(opt)
                  .CaseLower("cuda", KernelType::Cuda)
                  .CaseLower("cublas", KernelType::Cublas)
                  .Default(std::nullopt);
  assert(type && "Invalid kernel type");

  return *type;
}
} // namespace

struct MatmulKernelGpu : public KernelInterface<CudaTensor<float>> {
  MatmulKernelGpu() : kernel(parseKernelOption(kernelType)) {
    isInit = cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS;
    if (!isInit)
      std::cerr << "CUBLAS initialization failed!\n";

    cublasSetStream(handle, stream);
  }

  ~MatmulKernelGpu() { cublasDestroy(handle); }

  void runCUDA(float *A, float *B, float *C, int m, int n, int k) {
    matmulCUDA(A, B, C, m, n, k);
    cudaError_t syncStatus = cudaDeviceSynchronize();
    if (syncStatus != cudaSuccess) {
      std::cerr << "cudaDeviceSynchronize error : cuda code=" << syncStatus
                << " - " << cudaGetErrorString(syncStatus) << "\n";
    }
  }

  void runCUBLAS(float *A, float *B, float *C, int m, int n, int k) {
    int lda = n;
    int ldb = k;
    int ldc = n;

    float alpha = 1.0f;
    float beta = 1.0f;

    cublasStatus_t gemmStatus =
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, A, lda,
                    B, ldb, &beta, C, ldc);
    if (gemmStatus != CUBLAS_STATUS_SUCCESS) {
      cudaError_t cudaStatus = cudaGetLastError();
      std::cerr << "cublasSgemm error : cublas code=" << gemmStatus
                << " cuda code=" << cudaStatus << " - "
                << cudaGetErrorString(cudaStatus) << "\n";
    }

    cudaError_t syncStatus = cudaDeviceSynchronize();
    if (syncStatus != cudaSuccess) {
      std::cerr << "cudaDeviceSynchronize error : cuda code=" << syncStatus
                << " - " << cudaGetErrorString(syncStatus) << "\n";
    }
  }

  void runRef(std::vector<CudaTensor<float>> &args) override {
    assert(args.size() == 3 && "wrong rank for MLP");
    assert(isInit && "Kernel failed initialization");

    if (!isInit)
      return;

    auto &a = args[0];
    auto &b = args[1];
    auto &o = args[2];

    // MATMUL O += A x B
    int m = o.tensor.getDim(0);
    int n = o.tensor.getDim(1);
    int k = a.tensor.getDim(1);

    switch (kernel) {
    case KernelType::Cuda: {
      float *A = a.gpuData;
      float *B = b.gpuData;
      float *C = o.gpuData;
      runCUDA(A, B, C, m, n, k);
      break;
    }
    case KernelType::Cublas: {
      // See:
      // https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication
      // Swap A and B, and col-major change m with n.
      float *A = b.gpuData;
      float *B = a.gpuData;
      float *C = o.gpuData;
      runCUBLAS(A, B, C, m, n, k);
      break;
    }
    }
  }

  cublasHandle_t handle;
  cudaStream_t stream = 0;
  bool isInit = false;
  KernelType kernel;
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

  CudaTensor<float> gpuA(std::move(SplatTensor<float>{{m, k}, 1}));
  CudaTensor<float> gpuB(std::move(SplatTensor<float>{{k, n}, 1}));
  CudaTensor<float> gpuC(std::move(SplatTensor<float>{{m, n}, 0}));

  if (!gpuA.initGpu() || !gpuB.initGpu() || !gpuC.initGpu()) {
    std::cerr << "Failed GPU tensor initialization\n";
    return 1;
  }

  double gflops = static_cast<double>(2 * n * m * k) / 1e9;
  auto bench =
      Benchmark<MatmulKernelGpu, CudaTensor<float>>(config.iter, gflops);
  std::vector<CudaTensor<float>> args;
  args.push_back(std::move(gpuA));
  args.push_back(std::move(gpuB));
  args.push_back(std::move(gpuC));
  bench.setArg(std::move(args));

  // Warmup
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
