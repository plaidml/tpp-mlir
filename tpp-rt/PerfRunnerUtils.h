//===- PerfRunnerUtils.h - Utils for debugging MLIR execution -------------===//
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

#ifndef TPP_EXECUTIONENGINE_PERFRUNNERUTILS_H
#define TPP_EXECUTIONENGINE_PERFRUNNERUTILS_H

#include "mlir/ExecutionEngine/RunnerUtils.h"

//===----------------------------------------------------------------------===//
// Benchamrk perf utils
//===----------------------------------------------------------------------===//

extern "C" MLIR_RUNNERUTILS_EXPORT int64_t _mlir_ciface_timer_alloc();

extern "C" MLIR_RUNNERUTILS_EXPORT void _mlir_ciface_timer_start(int64_t);

extern "C" MLIR_RUNNERUTILS_EXPORT void _mlir_ciface_timer_stop(int64_t);

extern "C" MLIR_RUNNERUTILS_EXPORT void _mlir_ciface_timer_accumulate(int64_t);

extern "C" MLIR_RUNNERUTILS_EXPORT double _mlir_ciface_timer_average(int64_t);

extern "C" MLIR_RUNNERUTILS_EXPORT double _mlir_ciface_timer_deviation(int64_t);

//===----------------------------------------------------------------------===//
// Perf dialect utils
//===----------------------------------------------------------------------===//

extern "C" MLIR_RUNNERUTILS_EXPORT int64_t _mlir_ciface_perf_start_timer();

extern "C" MLIR_RUNNERUTILS_EXPORT double _mlir_ciface_perf_stop_timer(int64_t);

extern "C" MLIR_RUNNERUTILS_EXPORT double
_mlir_ciface_perf_mean(UnrankedMemRefType<double> *);

extern "C" MLIR_RUNNERUTILS_EXPORT double
_mlir_ciface_perf_stdev(UnrankedMemRefType<double> *, double);

extern "C" MLIR_RUNNERUTILS_EXPORT void
_mlir_ciface_perf_sink_memref_i8(UnrankedMemRefType<int8_t> *);
extern "C" MLIR_RUNNERUTILS_EXPORT void
_mlir_ciface_perf_sink_memref_i16(UnrankedMemRefType<int16_t> *);
extern "C" MLIR_RUNNERUTILS_EXPORT void
_mlir_ciface_perf_sink_memref_i32(UnrankedMemRefType<int32_t> *);
extern "C" MLIR_RUNNERUTILS_EXPORT void
_mlir_ciface_perf_sink_memref_i64(UnrankedMemRefType<int64_t> *);

extern "C" MLIR_RUNNERUTILS_EXPORT void _mlir_ciface_perf_sink_i8(int8_t);
extern "C" MLIR_RUNNERUTILS_EXPORT void _mlir_ciface_perf_sink_i16(int16_t);
extern "C" MLIR_RUNNERUTILS_EXPORT void _mlir_ciface_perf_sink_i32(int32_t);
extern "C" MLIR_RUNNERUTILS_EXPORT void _mlir_ciface_perf_sink_i64(int64_t);

extern "C" MLIR_RUNNERUTILS_EXPORT void _mlir_ciface_perf_sink_f32(float);
extern "C" MLIR_RUNNERUTILS_EXPORT void _mlir_ciface_perf_sink_f64(double);
#endif // TPP_EXECUTIONENGINE_PERFRUNNERUTILS_H
