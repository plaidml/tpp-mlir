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
#include <chrono>
#include <ctime>

namespace {
/// Local class to keep the results and stats of each measurement
class PerfResults {
  /// Mean
  double mean = 0.0;
  /// Standard deviation
  double stdev = 0.0;
  /// Start (reset by accumulate)
  std::chrono::high_resolution_clock::time_point start;
  /// Stop (reset by accumulate)
  std::chrono::high_resolution_clock::time_point stop;
  /// Vector with the timings
  std::vector<double> timings;
  /// Locked timings
  bool locked = false;

  /// Generate stats, locks the struct, can't collect stats any more
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
    locked = true;
  }

  /// Return true if time_point hasn't been used yet
  bool isZero(std::chrono::high_resolution_clock::time_point point) {
    return point.time_since_epoch().count() == 0;
  }

  /// Zero a time_point
  void zero(std::chrono::high_resolution_clock::time_point& point) {
    point = std::chrono::high_resolution_clock::time_point();
  }

public:
  /// Starts the timer
  void startTimer() {
    assert(!locked && "Start called after stats produced");
    assert(isZero(start) && "Start called twice");
    start = std::chrono::high_resolution_clock::now();
  }

  /// Stops the timer, accumulates, clears state
  void stopTimer() {
    assert(!locked && "Stop called after stats produced");
    assert(!isZero(start) && "Stop called before start");
    assert(isZero(stop) && "Stop called twice");
    stop = std::chrono::high_resolution_clock::now();
    auto val =
        std::chrono::duration_cast<std::chrono::duration<double>>(stop - start)
            .count();
    timings.push_back(val);
    zero(start);
    zero(stop);
  }

  /// Get mean of timings. Locks the timer, only calculate stats once.
  double getMean() {
    if (!locked) {
      assert(isZero(start) && isZero(stop) && "Mismatch call to start/stop");
      stats();
    }
    return mean;
  }

  /// Get stdev of timings. Locks the timer, only calculate stats once.
  double getStdev() {
    if (!locked) {
      assert(isZero(start) && isZero(stop) && "Mismatch call to start/stop");
      stats();
    }
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

void _mlir_ciface_timer_start(int64_t acc) {
  assert(acc >= 0 && (int64_t)timerResults.size() > acc && "Invalid timer ID");
  auto& perfResults = timerResults[acc];
  perfResults.startTimer();
}

void _mlir_ciface_timer_stop(int64_t acc) {
  assert(acc >= 0 && (int64_t)timerResults.size() > acc && "Invalid timer ID");
  auto& perfResults = timerResults[acc];
  perfResults.stopTimer();
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

