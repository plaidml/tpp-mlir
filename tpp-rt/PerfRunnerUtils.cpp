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
  std::chrono::system_clock::time_point start{
      std::chrono::system_clock::duration{startTimestamp}};
  return std::chrono::duration_cast<std::chrono::duration<double>>(stop - start)
      .count();
}

// A generic sink function.
// Its aim is to ensure that the passed data and its producers cannot be
// optimized away such that the time measured by a benchmark loop correctly
// represents the full workload.
static void __attribute__((optnone)) perf_sink(void *data) { (void)data; }

/*
  Perf dialect runtime bindings for common perf.sink op argument types.
*/
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
