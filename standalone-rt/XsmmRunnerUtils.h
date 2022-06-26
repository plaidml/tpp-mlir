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
_mlir_ciface_xsmm_matmul_invoke(int64_t id, UnrankedMemRefType<float> *A,
                                UnrankedMemRefType<float> *B,
                                UnrankedMemRefType<float> *C);

extern "C" MLIR_RUNNERUTILS_EXPORT int64_t _mlir_ciface_xsmm_matmul_dispatch(
    int32_t lda, int32_t ldb, int32_t ldc, int32_t m, int32_t n, int32_t k);

extern "C" MLIR_RUNNERUTILS_EXPORT int64_t _mlir_ciface_xsmm_unary_dispatch(
    int32_t m, int32_t n, int32_t ldi, int32_t ldo, int32_t in_type,
    int32_t compute_type, int32_t out_type, int32_t type, int32_t bcast_type);

extern "C" MLIR_RUNNERUTILS_EXPORT int64_t
_mlir_ciface_xsmm_unary_invoke(int64_t addr, void *input, void *output);

#endif // STANDALONE_EXECUTIONENGINE_CRUNNERUTILS_H
