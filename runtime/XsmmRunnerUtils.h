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

extern "C" MLIR_RUNNERUTILS_EXPORT int64_t
xsmm_gemm_dispatch(const libxsmm_datatype, int64_t, int64_t, int64_t, int64_t,
                   int64_t, int64_t, const libxsmm_gemm_flags);

extern "C" MLIR_RUNNERUTILS_EXPORT int64_t xsmm_unary_dispatch(
    const libxsmm_meltw_unary_type, const libxsmm_datatype, int64_t, int64_t,
    int64_t, int64_t, const libxsmm_meltw_unary_flags);

extern "C" MLIR_RUNNERUTILS_EXPORT int64_t xsmm_binary_dispatch(
    const libxsmm_meltw_binary_type, const libxsmm_datatype, int64_t, int64_t,
    int64_t, int64_t, int64_t, const libxsmm_meltw_binary_flags);

extern "C" MLIR_RUNNERUTILS_EXPORT int64_t
xsmm_brgemm_dispatch(const libxsmm_datatype, int64_t, int64_t, int64_t, int64_t,
                     int64_t, int64_t, const libxsmm_gemm_flags);

extern "C" MLIR_RUNNERUTILS_EXPORT void
xsmm_gemm_invoke(const libxsmm_datatype dType, int64_t addr, int64_t rankA,
                 void *memrefDescA, int64_t rankB, void *memrefDescB,
                 int64_t rankC, void *memrefDescC);

//TODO: Remove this function as all unary ops are expected to work in place.
extern "C" MLIR_RUNNERUTILS_EXPORT void
xsmm_unary_invoke(const libxsmm_datatype dType, int64_t addr, int64_t inputRank,
                  void *inputMemrefDesc, int64_t outputRank,
                  void *outputMemrefDesc);

extern "C" MLIR_RUNNERUTILS_EXPORT void
xsmm_binary_invoke(const libxsmm_datatype dType, int64_t addr, int64_t lhsRank,
                   void *lhsMemrefDesc, int64_t rhsRank, void *rhsMemrefDesc,
                   int64_t outRank, void *outMemrefDesc);

extern "C" MLIR_RUNNERUTILS_EXPORT void
xsmm_unary_invoke_inline(const libxsmm_datatype, int64_t, int64_t, void *);

extern "C" MLIR_RUNNERUTILS_EXPORT void
xsmm_unary_scalar_invoke(const libxsmm_datatype dType, int64_t addr,
                         float input, int64_t rankOutput,
                         void *outputMemrefDesc);

extern "C" MLIR_RUNNERUTILS_EXPORT void
xsmm_brgemm_invoke(const libxsmm_datatype dType, int64_t addr, int64_t rankA,
                   void *memrefDescA, int64_t rankB, void *memrefDescB,
                   int64_t rankC, void *memrefDescC, int64_t numBatches);

extern "C" MLIR_RUNNERUTILS_EXPORT void
xsmm_fused_brgemm_invoke(const libxsmm_datatype dType, int64_t addr,
                         int64_t rankA, void *memrefDescA, int64_t rankB,
                         void *memrefDescB, int64_t rankC, void *memrefDescC,
                         int64_t rankD, void *memrefDescD, int64_t numBatches);

//----------------------------------------------------------------------------//
// BRGEMM connection on the IREE side.
//----------------------------------------------------------------------------//

/// Eternal functions imported in IREE must pass everything via void*.
extern "C" MLIR_RUNNERUTILS_EXPORT int
iree_xsmm_brgemm_dispatch(void *context, void *params, void *reserved);
extern "C" MLIR_RUNNERUTILS_EXPORT int
iree_xsmm_gemm_dispatch(void *context, void *params, void *reserved);
extern "C" MLIR_RUNNERUTILS_EXPORT int
iree_xsmm_unary_dispatch(void *context, void *params, void *reserved);
extern "C" MLIR_RUNNERUTILS_EXPORT int
iree_xsmm_binary_dispatch(void *context, void *params, void *reserved);

extern "C" MLIR_RUNNERUTILS_EXPORT int
iree_xsmm_brgemm_invoke(void *context, void *params, void *reserved);
extern "C" MLIR_RUNNERUTILS_EXPORT int
iree_xsmm_gemm_invoke(void *context, void *params, void *reserved);
extern "C" MLIR_RUNNERUTILS_EXPORT int
iree_xsmm_unary_invoke(void *context, void *params, void *reserved);
extern "C" MLIR_RUNNERUTILS_EXPORT int
iree_xsmm_binary_invoke(void *context, void *params, void *reserved);

#endif // TPP_EXECUTIONENGINE_CRUNNERUTILS_H
