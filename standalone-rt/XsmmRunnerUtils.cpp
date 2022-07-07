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

  float *addr_a = matrixA.data + matrixA.offset;
  float *addr_b = matrixB.data + matrixB.offset;
  float *addr_c = matrixC.data + matrixC.offset;
  /*
    int64_t M = matrixC.sizes[0];
    int64_t N = matrixC.sizes[1];
    int64_t K = matrixA.sizes[1];

    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        for (int k = 0; k < K; k++) {
          float *curr_addr_a = i * matrixA.strides[0] + addr_a + k;
          float *curr_addr_b = k * matrixB.strides[0] + addr_b + j;
          float *curr_addr_c = i * matrixC.strides[0] + addr_c + j;
          *curr_addr_c += (*curr_addr_a) * (*curr_addr_b);
        }
      }
    }
  */
  libxsmm_xmmfunction sgemm;
  libxsmm_gemm_param gemm_param;
  // LIBXSMM col-major change A with B.
  gemm_param.a.primary = (void *)addr_b;
  gemm_param.b.primary = (void *)addr_a;
  gemm_param.c.primary = (void *)addr_c;
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
  l_shape.lda = ldb;
  l_shape.ldb = lda;
  l_shape.ldc = ldc;
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

  // std::cout << "ldi: " << ldi << "\n";
  // std::cout << "ldo: " << ldo << "\n";
  // std::cout << "m: " << m << "\n";
  // std::cout << "n: " << n << "\n";
  // std::cout << "type: " << type << "\n";
  // std::cout << "bcast_type: " << bcast_type << "\n";

  libxsmm_meltw_unary_flags unary_flags =
      static_cast<libxsmm_meltw_unary_flags>(bcast_type);

  libxsmm_meltw_unary_shape unary_shape;

  // Row major to col major swap m with n.
  unary_shape.m = static_cast<libxsmm_blasint>(n);
  unary_shape.n = static_cast<libxsmm_blasint>(m);
  unary_shape.in0_type = LIBXSMM_DATATYPE_F32;
  unary_shape.comp_type = LIBXSMM_DATATYPE_F32;
  unary_shape.out_type = LIBXSMM_DATATYPE_F32;
  unary_shape.ldi = static_cast<libxsmm_blasint>(ldi);
  unary_shape.ldo = static_cast<libxsmm_blasint>(ldo);

  libxsmm_meltwfunction_unary kernel = libxsmm_dispatch_meltw_unary_v2(
      static_cast<libxsmm_meltw_unary_type>(type), unary_shape,
      static_cast<libxsmm_bitfield>(unary_flags));

  return reinterpret_cast<int64_t>(kernel);
}

extern "C" int64_t _mlir_ciface_xsmm_binary_dispatch(int64_t m, int64_t n,
                                                     int64_t ldiLhs,
                                                     int64_t ldiRhs,
                                                     int64_t ldo, int64_t type,
                                                     int64_t bcast_type) {

  libxsmm_meltw_binary_flags binary_flags =
      static_cast<libxsmm_meltw_binary_flags>(bcast_type);

  libxsmm_meltw_binary_shape binary_shape;

  // Row major to col major swap m with n.
  binary_shape.m = static_cast<libxsmm_blasint>(n);
  binary_shape.n = static_cast<libxsmm_blasint>(m);
  binary_shape.in0_type = LIBXSMM_DATATYPE_F32;
  binary_shape.in1_type = LIBXSMM_DATATYPE_F32;
  binary_shape.comp_type = LIBXSMM_DATATYPE_F32;
  binary_shape.out_type = LIBXSMM_DATATYPE_F32;
  binary_shape.ldi = static_cast<libxsmm_blasint>(ldiLhs);
  binary_shape.ldi2 = static_cast<libxsmm_blasint>(ldiRhs);
  binary_shape.ldo = static_cast<libxsmm_blasint>(ldo);

  libxsmm_meltwfunction_binary kernel = libxsmm_dispatch_meltw_binary_v2(
      static_cast<libxsmm_meltw_binary_type>(type), binary_shape,
      static_cast<libxsmm_bitfield>(binary_flags));

  return reinterpret_cast<int64_t>(kernel);
}

extern "C" void
_mlir_ciface_xsmm_unary_invoke(int64_t addr, UnrankedMemRefType<float> *input,
                               UnrankedMemRefType<float> *output) {
  // std::cout << "tensor input: \n";
  // printMemRefMetaData(std::cout, DynamicMemRefType<float>(*input));
  // std::cout << "tensor output: \n";
  // printMemRefMetaData(std::cout, DynamicMemRefType<float>(*output));

  DynamicMemRefType<float> tensorA = DynamicMemRefType<float>(*input);
  DynamicMemRefType<float> tensorB = DynamicMemRefType<float>(*output);

  libxsmm_meltwfunction_unary kernel =
      reinterpret_cast<libxsmm_meltwfunction_unary>(addr);
  libxsmm_meltw_unary_param param;
  param.in.primary = (void *)tensorA.data;
  param.out.primary = (void *)tensorB.data;
  kernel(&param);
}

extern "C" void
_mlir_ciface_xsmm_binary_invoke(int64_t addr, UnrankedMemRefType<float> *lhs,
                                UnrankedMemRefType<float> *rhs,
                                UnrankedMemRefType<float> *output) {

  DynamicMemRefType<float> tensorLhs = DynamicMemRefType<float>(*lhs);
  DynamicMemRefType<float> tensorRhs = DynamicMemRefType<float>(*rhs);
  DynamicMemRefType<float> tensorOut = DynamicMemRefType<float>(*output);

  float *addr_tensor_lhs = tensorLhs.data + tensorLhs.offset;
  float *addr_tensor_rhs = tensorRhs.data + tensorRhs.offset;
  float *addr_tensor_out = tensorOut.data + tensorOut.offset;

  libxsmm_meltwfunction_binary kernel =
      reinterpret_cast<libxsmm_meltwfunction_binary>(addr);
  libxsmm_meltw_binary_param param;
  // TODO: check if we need to swap also here.
  param.in0.primary = (void *)addr_tensor_lhs;
  param.in1.primary = (void *)addr_tensor_rhs;
  param.out.primary = (void *)addr_tensor_out;
  kernel(&param);
}

extern "C" void
_mlir_ciface_xsmm_unary_scalar_invoke(int64_t addr, float input,
                                      UnrankedMemRefType<float> *output) {

  DynamicMemRefType<float> tensorB = DynamicMemRefType<float>(*output);

  libxsmm_meltwfunction_unary kernel =
      reinterpret_cast<libxsmm_meltwfunction_unary>(addr);
  libxsmm_meltw_unary_param param;
  param.in.primary = (void *)&input;
  param.out.primary = (void *)tensorB.data;
  kernel(&param);
}
