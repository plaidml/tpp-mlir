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

#include "XsmmRunnerUtils.h"
#include "libxsmm.h" // NOLINT [build/include_subdir]

extern "C" void _mlir_ciface_xsmm_gemm_invoke(int64_t funcAddr,
                                              UnrankedMemRefType<double> *A,
                                              UnrankedMemRefType<double> *B,
                                              UnrankedMemRefType<double> *C) {
  std::cout << "matrix A: \n";
  printMemRefMetaData(std::cout, DynamicMemRefType<double>(*A));
  std::cout << "\n";
  std::cout << "matrix B: \n";
  printMemRefMetaData(std::cout, DynamicMemRefType<double>(*B));
  std::cout << "\n";
  std::cout << "matrix C: \n";
  printMemRefMetaData(std::cout, DynamicMemRefType<double>(*C));
  std::cout << "\n";

  DynamicMemRefType<double> matrixA = DynamicMemRefType<double>(*A);
  DynamicMemRefType<double> matrixB = DynamicMemRefType<double>(*B);
  DynamicMemRefType<double> matrixC = DynamicMemRefType<double>(*C);

  libxsmm_xmmfunction sgemm;
  libxsmm_gemm_param gemm_param;
  gemm_param.a.primary = (void *)matrixA.basePtr;
  gemm_param.b.primary = (void *)matrixB.basePtr;
  gemm_param.c.primary = (void *)matrixC.basePtr;
  sgemm.gemm = reinterpret_cast<libxsmm_gemmfunction>(funcAddr);
  sgemm.gemm(&gemm_param);
}

extern "C" int64_t _mlir_ciface_xsmm_gemm_dispatch(int32_t lda, int32_t ldb,
                                                   int32_t ldc, int32_t m,
                                                   int32_t n, int32_t k) {
  std::cout << "lda: " << lda << "\n";
  std::cout << "ldb: " << ldb << "\n";
  std::cout << "ldc: " << ldc << "\n";
  std::cout << "m: " << m << "\n";
  std::cout << "n: " << n << "\n";
  std::cout << "k: " << k << "\n";

  libxsmm_blasint lda_int = lda;
  libxsmm_blasint ldb_int = ldb;
  libxsmm_blasint ldc_int = ldc;
  libxsmm_blasint m_int = m;
  libxsmm_blasint n_int = n;
  libxsmm_blasint k_int = k;

  libxsmm_gemm_shape l_shape;
  libxsmm_bitfield l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  libxsmm_bitfield l_prefetch_flags = 0;

  l_shape.m = n_int;
  l_shape.n = m_int;
  l_shape.k = k_int;
  l_shape.lda = ldb_int;
  l_shape.ldb = lda_int;
  l_shape.ldc = ldc_int;
  l_shape.a_in_type = LIBXSMM_DATATYPE_F32;
  l_shape.b_in_type = LIBXSMM_DATATYPE_F32;
  l_shape.out_type = LIBXSMM_DATATYPE_F32;
  l_shape.comp_type = LIBXSMM_DATATYPE_F32;

  auto sgemm = libxsmm_dispatch_gemm_v2(l_shape, l_flags, l_prefetch_flags);

  return reinterpret_cast<int64_t>(sgemm);
}
