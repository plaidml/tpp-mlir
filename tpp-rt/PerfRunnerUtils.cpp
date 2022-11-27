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

#include <chrono>
#include <ctime>

#include "PerfRunnerUtils.h"
#include "Utils/Perf.h"

/// Vector with all results
/// Using memref/vector in MLIR is too much of a pain.
static std::vector<PerfResults> timerResults;

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

