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

extern "C" void
_mlir_ciface_linalg_matmul_view64x64xf32_view64x64xf32_view64x64xf32(
    StridedMemRefType<float, 2> *A, StridedMemRefType<float, 2> *B,
    StridedMemRefType<float, 2> *C) {

  // printMemRefMetaData(std::cout, *A);
  // printMemRefMetaData(std::cout, *B);
  // printMemRefMetaData(std::cout, *C);

  dnnl_dim_t dim = 64;
  auto status =
      dnnl_sgemm('n', 'n', dim, dim, dim, 1.0, A->data + A->offset, dim,
                 B->data + B->offset, dim, 1.0, C->data + C->offset, dim);
  assert(status == 0);
}
