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

#include "mlir/ExecutionEngine/Float16bits.h"
#include "mlir/ExecutionEngine/RunnerUtils.h"

// TODO: here we want to have dispatch/invoke only for unary/binary and ternary.
// matmul, brgemm are way too specific.
extern "C" MLIR_RUNNERUTILS_EXPORT void
_mlir_ciface_xsmm_matmul_invoke_f32(int64_t, UnrankedMemRefType<float> *,
                                    UnrankedMemRefType<float> *,
                                    UnrankedMemRefType<float> *);

extern "C" MLIR_RUNNERUTILS_EXPORT
    int64_t _mlir_ciface_xsmm_matmul_dispatch_f32(int64_t, int64_t, int64_t,
                                                  int64_t, int64_t, int64_t);

extern "C" MLIR_RUNNERUTILS_EXPORT void
_mlir_ciface_xsmm_matmul_invoke_bf16(int64_t, UnrankedMemRefType<bf16> *,
                                     UnrankedMemRefType<bf16> *,
                                     UnrankedMemRefType<bf16> *);

extern "C" MLIR_RUNNERUTILS_EXPORT
    int64_t _mlir_ciface_xsmm_matmul_dispatch_bf16(int64_t, int64_t, int64_t,
                                                   int64_t, int64_t, int64_t);

extern "C" MLIR_RUNNERUTILS_EXPORT
    int64_t _mlir_ciface_xsmm_unary_dispatch(int64_t, int64_t, int64_t, int64_t,
                                             int64_t, int64_t);

extern "C" MLIR_RUNNERUTILS_EXPORT int64_t _mlir_ciface_xsmm_binary_dispatch(
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);

extern "C" MLIR_RUNNERUTILS_EXPORT
    int64_t _mlir_ciface_xsmm_brgemm_dispatch(int64_t, int64_t, int64_t,
                                              int64_t, int64_t, int64_t);

extern "C" MLIR_RUNNERUTILS_EXPORT void
_mlir_ciface_xsmm_unary_invoke(int64_t, UnrankedMemRefType<float> *,
                               UnrankedMemRefType<float> *);

extern "C" MLIR_RUNNERUTILS_EXPORT void
_mlir_ciface_xsmm_binary_invoke(int64_t, UnrankedMemRefType<float> *,
                                UnrankedMemRefType<float> *,
                                UnrankedMemRefType<float> *);

extern "C" MLIR_RUNNERUTILS_EXPORT void
_mlir_ciface_xsmm_unary_scalar_invoke(int64_t, float,
                                      UnrankedMemRefType<float> *);

extern "C" MLIR_RUNNERUTILS_EXPORT void
_mlir_ciface_matrix_copy_NC_to_NCNC(UnrankedMemRefType<float> *,
                                    UnrankedMemRefType<float> *, int64_t,
                                    int64_t, int64_t, int64_t);

extern "C" MLIR_RUNNERUTILS_EXPORT void
_mlir_ciface_xsmm_brgemm_invoke(int64_t, UnrankedMemRefType<float> *,
                                UnrankedMemRefType<float> *,
                                UnrankedMemRefType<float> *, int64_t);

//----------------------------------------------------------------------------//
// BRGEMM connection on the IREE side.
//----------------------------------------------------------------------------//

/// Eternal functions imported in IREE must pass everything via void*.
extern "C" MLIR_RUNNERUTILS_EXPORT int
iree_xsmm_brgemm_dispatch_f32(void *params);
extern "C" MLIR_RUNNERUTILS_EXPORT int
iree_xsmm_matmul_dispatch_f32(void *params);
extern "C" MLIR_RUNNERUTILS_EXPORT int iree_xsmm_unary_dispatch(void *params);

extern "C" MLIR_RUNNERUTILS_EXPORT int
iree_xsmm_brgemm_invoke_f32(void *params);
extern "C" MLIR_RUNNERUTILS_EXPORT int
iree_xsmm_matmul_invoke_f32(void *params);
extern "C" MLIR_RUNNERUTILS_EXPORT int iree_xsmm_unary_invoke(void *params);

#endif // STANDALONE_EXECUTIONENGINE_CRUNNERUTILS_H
