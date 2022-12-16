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
#include <cmath>
#include <ctime>

#include "PerfRunnerUtils.h"
#include "Utils/Perf.h"

/// Vector with all results
/// Using memref/vector in MLIR is too much of a pain.
static std::vector<PerfResults> timerResults;

//===----------------------------------------------------------------------===//
// Benchamrk perf utils
//===----------------------------------------------------------------------===//

/// Returns the index of the result in the local vector that can be
/// used as an ID to time, accumulate and get stats.
int64_t _mlir_ciface_timer_alloc() {
  timerResults.push_back({});
  return timerResults.size() - 1;
}

void _mlir_ciface_timer_start(int64_t acc) {
  assert(acc >= 0 && (int64_t)timerResults.size() > acc && "Invalid timer ID");
  auto &perfResults = timerResults[acc];
  perfResults.startTimer();
}

void _mlir_ciface_timer_stop(int64_t acc) {
  assert(acc >= 0 && (int64_t)timerResults.size() > acc && "Invalid timer ID");
  auto &perfResults = timerResults[acc];
  perfResults.stopTimer();
}

double _mlir_ciface_timer_average(int64_t acc) {
  assert(acc >= 0 && (int64_t)timerResults.size() > acc && "Invalid timer ID");
  auto &perfResults = timerResults[acc];
  return perfResults.getMean();
}

double _mlir_ciface_timer_deviation(int64_t acc) {
  assert(acc >= 0 && (int64_t)timerResults.size() > acc && "Invalid timer ID");
  auto &perfResults = timerResults[acc];
  return perfResults.getStdev();
}

//===----------------------------------------------------------------------===//
// Perf dialect utils
//===----------------------------------------------------------------------===//

int64_t _mlir_ciface_perf_start_timer() {
  auto timer = _mlir_ciface_timer_alloc();
  _mlir_ciface_timer_start(timer);
  return timer;
}

double _mlir_ciface_perf_stop_timer(int64_t timer) {
  assert(timer >= 0 && (int64_t)timerResults.size() > timer &&
         "Invalid timer ID");
  auto &perfResults = timerResults[timer];
  return perfResults.stopTimer();
}

double _mlir_ciface_perf_mean(UnrankedMemRefType<double> *deltasBuff) {
  auto deltas = DynamicMemRefType<double>(*deltasBuff);
  assert(deltas.rank == 1 && "Invalid deltas buffer");

  const int size = deltas.sizes[0];
  double sum = 0.0;
  for (auto it = deltas.begin(); it != deltas.end(); ++it)
    sum += *it;
  return sum / size;
}

double _mlir_ciface_perf_stdev(UnrankedMemRefType<double> *deltasBuff,
                               double mean) {
  auto deltas = DynamicMemRefType<double>(*deltasBuff);
  assert(deltas.rank == 1 && "Invalid deltas buffer");

  const int size = deltas.sizes[0];
  double sum = 0.0;
  for (auto it = deltas.begin(); it != deltas.end(); ++it) {
    double delta = *it - mean;
    sum += delta * delta;
  }
  return std::sqrt(sum / size);
}

static void __attribute__((optnone)) perf_sink(void *data) { (void)data; }

void _mlir_ciface_perf_sink_memref_i8(UnrankedMemRefType<int8_t> *val) {
  perf_sink((void *)&val);
}
void _mlir_ciface_perf_sink_memref_i16(UnrankedMemRefType<int16_t> *val) {
  perf_sink((void *)&val);
}
void _mlir_ciface_perf_sink_memref_i32(UnrankedMemRefType<int32_t> *val) {
  perf_sink((void *)&val);
}
void _mlir_ciface_perf_sink_memref_i64(UnrankedMemRefType<int64_t> *val) {
  perf_sink((void *)&val);
}

void _mlir_ciface_perf_sink_i8(int8_t val) { perf_sink((void *)&val); }
void _mlir_ciface_perf_sink_i16(int16_t val) { perf_sink((void *)&val); }
void _mlir_ciface_perf_sink_i32(int32_t val) { perf_sink((void *)&val); }
void _mlir_ciface_perf_sink_i64(int64_t val) { perf_sink((void *)&val); }

void _mlir_ciface_perf_sink_f32(float val) { perf_sink((void *)&val); }
void _mlir_ciface_perf_sink_f64(double val) { perf_sink((void *)&val); }
