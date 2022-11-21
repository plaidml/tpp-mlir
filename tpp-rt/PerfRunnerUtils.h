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

extern "C" MLIR_RUNNERUTILS_EXPORT int64_t _mlir_ciface_timer_alloc();

extern "C" MLIR_RUNNERUTILS_EXPORT void _mlir_ciface_timer_start(int64_t);

extern "C" MLIR_RUNNERUTILS_EXPORT void _mlir_ciface_timer_stop(int64_t);

extern "C" MLIR_RUNNERUTILS_EXPORT void _mlir_ciface_timer_accumulate(int64_t);

extern "C" MLIR_RUNNERUTILS_EXPORT double _mlir_ciface_timer_average(int64_t);

extern "C" MLIR_RUNNERUTILS_EXPORT double _mlir_ciface_timer_deviation(int64_t);

#endif // TPP_EXECUTIONENGINE_PERFRUNNERUTILS_H
