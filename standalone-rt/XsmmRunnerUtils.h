//===- CRunnerUtils.h - Utils for debugging MLIR execution ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares basic classes and functions to manipulate structured MLIR
// types at runtime. Entities in this file must be compliant with C++11 and be
// retargetable, including on targets without a C++ runtime.
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_EXECUTIONENGINE_CRUNNERUTILS_H
#define STANDALONE_EXECUTIONENGINE_CRUNNERUTILS_H

#include "mlir/ExecutionEngine/RunnerUtils.h"

extern "C" MLIR_RUNNERUTILS_EXPORT void
_mlir_ciface_xsmm_matmul_invoke(int64_t, UnrankedMemRefType<float> *,
                                UnrankedMemRefType<float> *,
                                UnrankedMemRefType<float> *);

extern "C" MLIR_RUNNERUTILS_EXPORT
    int64_t _mlir_ciface_xsmm_matmul_dispatch(int32_t, int32_t, int32_t,
                                              int32_t, int32_t, int32_t);

extern "C" MLIR_RUNNERUTILS_EXPORT
    int64_t _mlir_ciface_xsmm_unary_dispatch(int32_t, int32_t, int32_t, int32_t,
                                             int32_t, int32_t, int32_t, int32_t,
                                             int32_t);

extern "C" MLIR_RUNNERUTILS_EXPORT void
_mlir_ciface_xsmm_unary_invoke(int64_t, UnrankedMemRefType<float> *,
                               UnrankedMemRefType<float> *);

#endif // STANDALONE_EXECUTIONENGINE_CRUNNERUTILS_H
