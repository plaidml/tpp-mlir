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

extern "C" void _mlir_ciface_xsmm_matmul_invoke(int64_t funcAddr,
                                                UnrankedMemRefType<float> *A,
                                                UnrankedMemRefType<float> *B,
                                                UnrankedMemRefType<float> *C) {

  // std::cout << "matrix A: \n";
  // printMemRefMetaData(std::cout, DynamicMemRefType<float>(*A));
  // std::cout << "\n";
  // std::cout << "matrix B: \n";
  // printMemRefMetaData(std::cout, DynamicMemRefType<float>(*B));
  // std::cout << "\n";
  // std::cout << "matrix C: \n";
  // printMemRefMetaData(std::cout, DynamicMemRefType<float>(*C));
  // std::cout << "\n";
  // std::cout << "funcAddr: " << funcAddr << "\n";

  DynamicMemRefType<float> matrixA = DynamicMemRefType<float>(*A);
  DynamicMemRefType<float> matrixB = DynamicMemRefType<float>(*B);
  DynamicMemRefType<float> matrixC = DynamicMemRefType<float>(*C);

  libxsmm_xmmfunction sgemm;
  libxsmm_gemm_param gemm_param;
  // LIBXSMM col-major change A with B.
  gemm_param.a.primary = (void *)matrixB.data;
  gemm_param.b.primary = (void *)matrixA.data;
  gemm_param.c.primary = (void *)matrixC.data;
  sgemm.gemm = reinterpret_cast<libxsmm_gemmfunction>(funcAddr);
  sgemm.gemm(&gemm_param);
}

extern "C" int64_t _mlir_ciface_xsmm_matmul_dispatch(int64_t m, int64_t n,
                                                     int64_t k, int64_t lda,
                                                     int64_t ldb, int64_t ldc) {
  // std::cout << "lda: " << lda << "\n";
  // std::cout << "ldb: " << ldb << "\n";
  // std::cout << "ldc: " << ldc << "\n";
  // std::cout << "m: " << m << "\n";
  // std::cout << "n: " << n << "\n";
  // std::cout << "k: " << k << "\n";

  libxsmm_blasint m_int = m;
  libxsmm_blasint n_int = n;
  libxsmm_blasint k_int = k;

  libxsmm_gemm_shape l_shape;
  libxsmm_bitfield l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  libxsmm_bitfield l_prefetch_flags = 0;

  // See:
  // https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication
  // LIBXSMM col-major change m with n.
  l_shape.m = n_int;
  l_shape.n = m_int;
  l_shape.k = k_int;
  l_shape.lda = n_int;
  l_shape.ldb = k_int;
  l_shape.ldc = n_int;
  l_shape.a_in_type = LIBXSMM_DATATYPE_F32;
  l_shape.b_in_type = LIBXSMM_DATATYPE_F32;
  l_shape.out_type = LIBXSMM_DATATYPE_F32;
  l_shape.comp_type = LIBXSMM_DATATYPE_F32;

  auto sgemm = libxsmm_dispatch_gemm_v2(l_shape, l_flags, l_prefetch_flags);
  return reinterpret_cast<int64_t>(sgemm);
}

extern "C" int64_t _mlir_ciface_xsmm_unary_dispatch(int64_t m, int64_t n,
                                                    int64_t ldi, int64_t ldo,
                                                    int64_t type,
                                                    int64_t bcast_type) {

  std::cout << "ldi: " << ldi << "\n";
  std::cout << "ldo: " << ldo << "\n";
  std::cout << "m: " << m << "\n";
  std::cout << "n: " << n << "\n";
  std::cout << "type: " << type << "\n";
  std::cout << "bcast_type: " << bcast_type << "\n";

  libxsmm_blasint ldi_int = ldi;
  libxsmm_blasint ldo_int = ldo;
  libxsmm_meltw_unary_flags unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;

  libxsmm_meltw_unary_shape unary_shape;

  unary_shape.m = static_cast<libxsmm_blasint>(n);
  unary_shape.n = static_cast<libxsmm_blasint>(m);
  unary_shape.in0_type = LIBXSMM_DATATYPE_F32;
  unary_shape.comp_type = LIBXSMM_DATATYPE_F32;
  unary_shape.out_type = LIBXSMM_DATATYPE_F32;
  unary_shape.ldi = ldi_int;
  unary_shape.ldo = ldo_int;

  libxsmm_meltwfunction_unary kernel = libxsmm_dispatch_meltw_unary_v2(
      static_cast<libxsmm_meltw_unary_type>(type), unary_shape,
      static_cast<libxsmm_bitfield>(unary_flags));

  return reinterpret_cast<int64_t>(kernel);
}

extern "C" void
_mlir_ciface_xsmm_unary_invoke(int64_t addr, UnrankedMemRefType<float> *input,
                               UnrankedMemRefType<float> *output) {
  std::cout << "tensor input: \n";
  printMemRefMetaData(std::cout, DynamicMemRefType<float>(*input));
  std::cout << "tensor output: \n";
  printMemRefMetaData(std::cout, DynamicMemRefType<float>(*output));

  DynamicMemRefType<float> tensorA = DynamicMemRefType<float>(*input);
  DynamicMemRefType<float> tensorB = DynamicMemRefType<float>(*output);

  libxsmm_meltwfunction_unary kernel =
      reinterpret_cast<libxsmm_meltwfunction_unary>(addr);
  libxsmm_meltw_unary_param param;
  param.in.primary = (void *)tensorA.data;
  param.out.primary = (void *)tensorB.data;
  kernel(&param);
}
