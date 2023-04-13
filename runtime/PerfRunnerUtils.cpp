//===- CRunnerUtils.cpp - Utils for MLIR execution ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities to measure performance (timers, statistics, etc).
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <chrono>
#include <ctime>

#include "PerfRunnerUtils.h"

//===----------------------------------------------------------------------===//
// Perf dialect utils
//===----------------------------------------------------------------------===//

// Return the current timestamp.
int64_t _mlir_ciface_perf_start_timer() {
  auto timestamp = std::chrono::high_resolution_clock::now();
  return timestamp.time_since_epoch().count();
}

// Compute time delta between the starting time and now.
double _mlir_ciface_perf_stop_timer(int64_t startTimestamp) {
  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::high_resolution_clock::time_point start{
      std::chrono::high_resolution_clock::duration{startTimestamp}};
  return std::chrono::duration_cast<std::chrono::duration<double>>(stop - start)
      .count();
}

double _mlir_ciface_perf_median(UnrankedMemRefType<double> *buf) {
  DynamicMemRefType<double> deltas(*buf);

  // First, sort the measurements.
  std::vector<double> data(deltas.begin(), deltas.end());
  std::sort(data.begin(), data.end());

  // Calculate median value.
  int64_t n = data.size();
  double median = 0.0f;
  if (n % 2 == 0) {
    median = (data[n / 2 - 1] + data[n / 2]) / 2.0;
  } else {
    median = data[n / 2];
  }

  return median;
}
