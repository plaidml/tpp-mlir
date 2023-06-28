//===- Bench.h - ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "Perf.h"
#include "Tensor.h"
#include <vector>

/// KernelInterface: Must be implemented by kernels to use Benchmark class.
///
/// T is a numeric type (int, float, double), and represents the payload
/// type of the input and output tensors, not necessarily the accumulation
/// and working internal tensors.
template <typename T> struct KernelInterface {
  virtual ~KernelInterface() {}
  /// Runs the reference kernel with the list of arguments.
  /// Output needs to be passed in the args list and is up to the user to
  /// implement the correct logic.
  virtual void runRef(std::vector<Tensor<T>> &args) = 0;

  /// Runs the XSMM kernel with the list of arguments.
  /// Output needs to be passed in the args list and is up to the user to
  /// implement the correct logic.
  virtual void runXSMM(std::vector<Tensor<T>> &args) = 0;
};

/// Benchmark: runs a kernel multiple times, takes timings, return avg.
///
/// Supports warm-up (JIT compilers, cache flush, etc), tensor initialization,
/// performance tracking, output comparison, etc.
///
/// Usage:
///  * Implement a kernel that implements the KernelInterface
///  * Define a variable as Benchmark<MyKernel>
template <class Kernel, class T> class Benchmark {
  /// Performance tracker, caches timings and calculates avg/dev
  PerfResults perf;
  /// The kernel to execute, must implement KernelInterface
  Kernel kernel;
  /// Cached arguments (and output) for the kernel call
  std::vector<Tensor<T>> args;
  /// Number of times to run
  size_t iter;
  /// Expected number of Giga FP ops in the kernel (for flops reporting)
  double gflops;
  /// Runs XSMM kernel or not.
  bool xsmm;

public:
  Benchmark(size_t iter, double gflops = 0.0, bool xsmm = false)
      : iter(iter), gflops(gflops), xsmm(xsmm) {}

  /// Sets argument (input/output) list.
  void setArg(std::vector<Tensor<T>> &&arg) { args = std::move(arg); }

  /// Inspects the tensors
  const Tensor<T> &getArg(size_t index) {
    assert(index < args.size() && "Invalid index for arguments");
    return args[index];
  }

  /// Warms up the kernel and populates the output (inside the args vector).
  void warmup() {
    if (xsmm)
      kernel.runXSMM(args);
    else
      kernel.runRef(args);
  }

  /// Runs the benchmark `iter` times and take timings
  void run() {
    // Easier to optimize / inline a branchless loop
    for (size_t i = 0; i < iter; i++) {
      perf.startTimer();
      if (xsmm)
        kernel.runXSMM(args);
      else
        kernel.runRef(args);
      perf.stopTimer();
    }
  }

  /// Returns the mean of the runs
  double getMean() {
    auto mean = perf.getMean();
    if (gflops) {
      return gflops / mean;
    }
    return mean;
  }

  /// Returns the standard deviation of the runs
  double getStdev() {
    auto stdev = perf.getStdev();
    if (gflops) {
      auto mean = perf.getMean();
      return (gflops * stdev) / (mean * mean);
    }
    return stdev;
  }
};
