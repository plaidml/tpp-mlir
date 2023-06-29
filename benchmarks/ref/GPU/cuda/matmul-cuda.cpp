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

template <class T> struct MatmulKernelCUBLAS : public KernelInterface<T> {
  void runRef(std::vector<T> &args) override {
    assert(args.size() == 3 && "wrong rank for MLP");
    auto &a = args[0];
    auto &b = args[1];
    auto &o = args[2];

    int m = o.tensor.getDim(0);
    int n = o.tensor.getDim(1);
    int k = a.tensor.getDim(1);

    // MATMUL O += A x B
    for (int mi = 0; mi < m; ++mi) {
      for (int ni = 0; ni < n; ++ni) {
        for (int ki = 0; ki < k; ++ki) {
          o.tensor[mi * n + ni] +=
              a.tensor[mi * k + ki] * b.tensor[ki * n + ni];
        }
      }
    }
  }
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

  ConstantTensor<float> matA({m, k});
  ConstantTensor<float> matB({k, n});
  EmptyTensor<float> matC({m, n});

  CudaTensor<float> gpuA(std::move(matA));
  CudaTensor<float> gpuB(std::move(matB));
  CudaTensor<float> gpuC(std::move(matC));

  if (!gpuA.initGpu() || !gpuB.initGpu() || !gpuC.initGpu())
    return 1;

  double gflops = static_cast<double>(2 * n * m * k) / 1e9;
  auto bench =
      Benchmark<MatmulKernelCUBLAS<CudaTensor<float>>, CudaTensor<float>>(
          config.iter, gflops);
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
