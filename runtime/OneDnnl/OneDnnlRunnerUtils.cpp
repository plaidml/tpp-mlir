//===- CRunnerUtils.cpp - Utils for MLIR execution ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements basic functions to manipulate structured MLIR types at
// runtime. Entities in this file are meant to be retargetable, including on
// targets without a C++ runtime, and must be kept C compatible.
//
//===----------------------------------------------------------------------===//

#include "OneDnnlRunnerUtils.h"
#include "dnnl.h"
#include <cassert>

extern "C" void linalg_matmul_blas(size_t m, size_t n, size_t k, const float *A,
                                   size_t offsetA, size_t lda, const float *B,
                                   size_t offsetB, size_t ldb, float *C,
                                   size_t offsetC, size_t ldc) {
  auto status = dnnl_sgemm('n', 'n', m, n, k, 1.0, A + offsetA, lda,
                           B + offsetB, ldb, 1.0, C + offsetC, ldc);
  assert(status == 0);
}
