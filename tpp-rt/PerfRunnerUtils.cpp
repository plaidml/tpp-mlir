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

#include "PerfRunnerUtils.h"
#include <cstdlib>
#include <time.h>

namespace {
/// Local class to keep the results and stats of each measurement
class PerfResults {
  double mean = 0.0;
  double stdev = 0.0;
  std::vector<double> timings;

  void stats() {
    auto size = timings.size();
    // Mean
    double sum = 0.0;
    for (size_t i = 0; i < size; i++)
      sum += timings[i];
    mean = sum / size;
    // Stdev
    sum = 0.0;
    for (size_t i = 0; i < size; i++) {
      double delta = timings[i] - mean;
      sum += delta * delta;
    }
    stdev = sqrt(sum / size);
  }

public:
  void accumulate(double val) {
    timings.push_back(val);
  }

  double getMean() {
    if (mean == 0.0)
      stats();
    return mean;
  }

  double getStdev() {
    if (stdev == 0.0)
      stats();
    return stdev;
  }
};

/// Vector with all results
/// Using memref/vector in MLIR is too much of a pain.
std::vector<PerfResults> timerResults;
} // namespace

/// Returns the index of the result in the local vector that can be
/// used as an ID to time, accumulate and get stats.
int64_t _mlir_ciface_timer_alloc() {
  timerResults.push_back({});
  return timerResults.size() - 1;
}

double _mlir_ciface_timer_now() { return (double)clock() / CLOCKS_PER_SEC; }

void _mlir_ciface_timer_accumulate(int64_t acc, double val) {
  assert(acc >= 0 && (int64_t)timerResults.size() > acc && "Invalid timer ID");
  auto& perfResults = timerResults[acc];
  perfResults.accumulate(val);
}

double _mlir_ciface_timer_average(int64_t acc) {
  assert(acc >= 0 && (int64_t)timerResults.size() > acc && "Invalid timer ID");
  auto& perfResults = timerResults[acc];
  return perfResults.getMean();
}

double _mlir_ciface_timer_deviation(int64_t acc) {
  assert(acc >= 0 && (int64_t)timerResults.size() > acc && "Invalid timer ID");
  auto& perfResults = timerResults[acc];
  return perfResults.getStdev();
}

