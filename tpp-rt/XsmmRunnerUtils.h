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

#ifndef TPP_EXECUTIONENGINE_CRUNNERUTILS_H
#define TPP_EXECUTIONENGINE_CRUNNERUTILS_H

#include "libxsmm.h"
#include "mlir/ExecutionEngine/Float16bits.h"
#include "mlir/ExecutionEngine/RunnerUtils.h"

// TODO: here we want to have dispatch/invoke only for unary/binary and ternary.
// matmul, brgemm are way too specific.
extern "C" MLIR_RUNNERUTILS_EXPORT void _mlir_ciface_xsmm_matmul_invoke(
    const libxsmm_datatype, int64_t, UnrankedMemRefType<char> *,
    UnrankedMemRefType<char> *, UnrankedMemRefType<char> *);

extern "C" MLIR_RUNNERUTILS_EXPORT int64_t
_mlir_ciface_xsmm_matmul_dispatch(const libxsmm_datatype, bool isVNNI, int64_t,
                                  int64_t, int64_t, int64_t, int64_t, int64_t);

extern "C" MLIR_RUNNERUTILS_EXPORT int64_t
_mlir_ciface_xsmm_unary_dispatch(const libxsmm_datatype, int64_t, int64_t,
                                 int64_t, int64_t, int64_t, int64_t);

extern "C" MLIR_RUNNERUTILS_EXPORT int64_t
_mlir_ciface_xsmm_binary_dispatch(const libxsmm_datatype, int64_t, int64_t,
                                  int64_t, int64_t, int64_t, int64_t, int64_t);

extern "C" MLIR_RUNNERUTILS_EXPORT int64_t
_mlir_ciface_xsmm_brgemm_dispatch(const libxsmm_datatype, bool isVNNI, int64_t,
                                  int64_t, int64_t, int64_t, int64_t, int64_t);

extern "C" MLIR_RUNNERUTILS_EXPORT int64_t
_mlir_ciface_xsmm_fused_brgemm_dispatch(const libxsmm_datatype, bool isVNNI,
                                        int64_t, int64_t, int64_t, int64_t,
                                        int64_t, int64_t);

//TODO: Remove this function as all unary ops are expected to work in place.
extern "C" MLIR_RUNNERUTILS_EXPORT void
_mlir_ciface_xsmm_unary_invoke(const libxsmm_datatype, int64_t,
                               UnrankedMemRefType<char> *,
                               UnrankedMemRefType<char> *);

extern "C" MLIR_RUNNERUTILS_EXPORT void _mlir_ciface_xsmm_binary_invoke(
    const libxsmm_datatype, int64_t, UnrankedMemRefType<char> *,
    UnrankedMemRefType<char> *, UnrankedMemRefType<char> *);

extern "C" MLIR_RUNNERUTILS_EXPORT void
_mlir_ciface_xsmm_unary_invoke_inline(const libxsmm_datatype, int64_t,
                                      UnrankedMemRefType<char> *);

extern "C" MLIR_RUNNERUTILS_EXPORT void
_mlir_ciface_xsmm_unary_scalar_invoke(const libxsmm_datatype, int64_t, float,
                                      UnrankedMemRefType<char> *);

extern "C" MLIR_RUNNERUTILS_EXPORT void _mlir_ciface_xsmm_brgemm_invoke(
    const libxsmm_datatype, int64_t, UnrankedMemRefType<char> *,
    UnrankedMemRefType<char> *, UnrankedMemRefType<char> *, int64_t);

extern "C" MLIR_RUNNERUTILS_EXPORT void _mlir_ciface_xsmm_fused_brgemm_invoke(
    const libxsmm_datatype, int64_t, UnrankedMemRefType<char> *,
    UnrankedMemRefType<char> *, UnrankedMemRefType<char> *,
    UnrankedMemRefType<char> *, int64_t);

//----------------------------------------------------------------------------//
// BRGEMM connection on the IREE side.
//----------------------------------------------------------------------------//

/// Eternal functions imported in IREE must pass everything via void*.
extern "C" MLIR_RUNNERUTILS_EXPORT int
iree_xsmm_brgemm_dispatch_f32(void *context, void *params, void *reserved);
extern "C" MLIR_RUNNERUTILS_EXPORT int
iree_xsmm_matmul_dispatch_f32(void *context, void *params, void *reserved);
extern "C" MLIR_RUNNERUTILS_EXPORT int
iree_xsmm_unary_dispatch(void *context, void *params, void *reserved);

extern "C" MLIR_RUNNERUTILS_EXPORT int
iree_xsmm_brgemm_invoke_f32(void *context, void *params, void *reserved);
extern "C" MLIR_RUNNERUTILS_EXPORT int
iree_xsmm_matmul_invoke_f32(void *context, void *params, void *reserved);
extern "C" MLIR_RUNNERUTILS_EXPORT int
iree_xsmm_unary_invoke(void *context, void *params, void *reserved);

#endif // TPP_EXECUTIONENGINE_CRUNNERUTILS_H
