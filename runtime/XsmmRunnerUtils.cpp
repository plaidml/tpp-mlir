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
#include "libxsmm_typedefs.h"

// Helper function prototypes.
static void printXsmmStruct(const libxsmm_gemm_shape &gemmShape,
                            FILE *outfile = stderr);
static void printXsmmStruct(const libxsmm_meltw_unary_shape &unaryShape,
                            FILE *outfile = stderr);
static void printXsmmStruct(const libxsmm_meltw_binary_shape &binaryShape,
                            FILE *outfile = stderr);
static void printXsmmStruct(const libxsmm_gemm_batch_reduce_config &brgemmShape,
                            FILE *outfile = stderr);

static bool isTransformUnary(const libxsmm_meltw_unary_type dtype) {
  switch (dtype) {
    // Zero
    case LIBXSMM_MELTW_TYPE_UNARY_XOR:
    // Copy
    case LIBXSMM_MELTW_TYPE_UNARY_IDENTITY:
    // Transpose
    case LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT:
    // VNNI2
    case LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2:
    case LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI2_TO_VNNI2T:
    case LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2T:
    case LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2_PAD:
    case LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADM_MOD2:
    case LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADN_MOD2:
    case LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD2:
    // VNNI4
    case LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4:
    case LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_VNNI4T:
    case LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4T:
    case LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4_PAD:
    case LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADM_MOD4:
    case LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADN_MOD4:
    case LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD4:
    case LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_NORM:
    case LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_VNNI2:
      return true;
    default:
      return false;
  }
}

extern "C" void _mlir_ciface_xsmm_matmul_invoke(const libxsmm_datatype dtype,
                                                int64_t funcAddr,
                                                UnrankedMemRefType<char> *A,
                                                UnrankedMemRefType<char> *B,
                                                UnrankedMemRefType<char> *C) {
  DynamicMemRefType<char> matrixA = DynamicMemRefType<char>(*A);
  DynamicMemRefType<char> matrixB = DynamicMemRefType<char>(*B);
  DynamicMemRefType<char> matrixC = DynamicMemRefType<char>(*C);

  //   std::cout << "matrix A: \n";
  //   printMemRefMetaData(std::cout, DynamicMemRefType<char>(*A));
  //    std::cout << "\n";
  //   std::cout << "matrix B: \n";
  //   printMemRefMetaData(std::cout, DynamicMemRefType<char>(*B));
  //   std::cout << "\n";
  //   std::cout << "matrix C: \n";
  //   printMemRefMetaData(std::cout, DynamicMemRefType<char>(*C));
  //   std::cout << "\n";
  //   std::cout << "funcAddr: " << funcAddr << "\n";

  //    int64_t M = matrixC.sizes[0];
  //    int64_t N = matrixC.sizes[1];
  //    int64_t K = matrixA.sizes[1];
  //
  //    for (int i = 0; i < M; i++) {
  //      for (int j = 0; j < N; j++) {
  //        for (int k = 0; k < K; k++) {
  //          float *curr_addr_a = i * matrixA.strides[0] + addr_a + k;
  //          float *curr_addr_b = k * matrixB.strides[0] + addr_b + j;
  //          float *curr_addr_c = i * matrixC.strides[0] + addr_c + j;
  //          *curr_addr_c += (*curr_addr_a) * (*curr_addr_b);
  //        }
  //      }
  //    }

  libxsmm_xmmfunction sgemm;
  libxsmm_gemm_param gemm_param;

  if (dtype == LIBXSMM_DATATYPE_F32) {
    float *addr_a = (float *)matrixA.data + matrixA.offset;
    float *addr_b = (float *)matrixB.data + matrixB.offset;
    float *addr_c = (float *)matrixC.data + matrixC.offset;
    // LIBXSMM col-major change A with B.
    gemm_param.a.primary = (void *)addr_b;
    gemm_param.b.primary = (void *)addr_a;
    gemm_param.c.primary = (void *)addr_c;

  } else if (dtype == LIBXSMM_DATATYPE_BF16) {
    bf16 *addr_a = (bf16 *)matrixA.data + matrixA.offset;
    bf16 *addr_b = (bf16 *)matrixB.data + matrixB.offset;
    bf16 *addr_c = (bf16 *)matrixC.data + matrixC.offset;
    //  LIBXSMM col-major change A with B.
    gemm_param.a.primary = (void *)addr_b;
    gemm_param.b.primary = (void *)addr_a;
    gemm_param.c.primary = (void *)addr_c;
  }
  sgemm.gemm = reinterpret_cast<libxsmm_gemmfunction>(funcAddr);
  sgemm.gemm(&gemm_param);
}

extern "C" int64_t _mlir_ciface_xsmm_matmul_dispatch(
    const libxsmm_datatype dtype, int64_t m, int64_t n, int64_t k, int64_t lda,
    int64_t ldb, int64_t ldc, const libxsmm_gemm_flags flags) {
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
  libxsmm_bitfield l_flags = LIBXSMM_GEMM_FLAG_NONE;
  if (flags == LIBXSMM_GEMM_FLAG_VNNI_B) {
    assert(dtype == LIBXSMM_DATATYPE_BF16);
    l_flags = l_flags | LIBXSMM_GEMM_FLAG_VNNI_A;
  } else {
    l_flags = l_flags | flags;
  }
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
  l_shape.a_in_type = dtype;
  l_shape.b_in_type = dtype;
  l_shape.out_type = dtype;
  // Retarget computation type from bf16 to f32 due to missing hardware support.
  l_shape.comp_type =
      dtype == LIBXSMM_DATATYPE_BF16 ? LIBXSMM_DATATYPE_F32 : dtype;

  auto sgemm = libxsmm_dispatch_gemm_v2(l_shape, l_flags, l_prefetch_flags);
  if (!sgemm) {
    fprintf(stderr, "failed to generate matmul func\n");
    fprintf(stderr, "dtype: %u\n", dtype);
    printXsmmStruct(l_shape);
    exit(-1);
  }

  return reinterpret_cast<int64_t>(sgemm);
}

extern "C" int64_t
_mlir_ciface_xsmm_unary_dispatch(const libxsmm_datatype dtype, int64_t m,
                                 int64_t n, int64_t ldi, int64_t ldo,
                                 int64_t type, int64_t bcast_type) {

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
  unary_shape.in0_type = dtype;
  // Retarget computation type from bf16 to f32 due to missing hardware support.
  // Copy and Zero should remain in BF16 to avoid useless up/down casts
  auto op_type = static_cast<libxsmm_meltw_unary_type>(type);
  auto force_fp32 =
      (dtype == LIBXSMM_DATATYPE_BF16 && isTransformUnary(op_type));
  unary_shape.comp_type = force_fp32 ? LIBXSMM_DATATYPE_F32 : dtype;
  unary_shape.out_type = dtype;
  unary_shape.ldi = static_cast<libxsmm_blasint>(ldi);
  unary_shape.ldo = static_cast<libxsmm_blasint>(ldo);

  libxsmm_meltwfunction_unary kernel = libxsmm_dispatch_meltw_unary_v2(
      static_cast<libxsmm_meltw_unary_type>(type), unary_shape,
      static_cast<libxsmm_bitfield>(unary_flags));
  if (!kernel) {
    fprintf(stderr, "failed to generate unary func\n");
    fprintf(stderr, "type: %ld\n", type);
    fprintf(stderr, "bcast_type: %ld\n", bcast_type);
    printXsmmStruct(unary_shape);
    exit(-1);
  }

  return reinterpret_cast<int64_t>(kernel);
}

extern "C" int64_t _mlir_ciface_xsmm_binary_dispatch(
    const libxsmm_datatype dtype, int64_t m, int64_t n, int64_t ldiLhs,
    int64_t ldiRhs, int64_t ldo, int64_t type, int64_t bcast_type) {

  libxsmm_meltw_binary_flags binary_flags =
      static_cast<libxsmm_meltw_binary_flags>(bcast_type);

  libxsmm_meltw_binary_shape binary_shape;

  // Row major to col major swap m with n.
  binary_shape.m = static_cast<libxsmm_blasint>(n);
  binary_shape.n = static_cast<libxsmm_blasint>(m);
  binary_shape.in0_type = dtype;
  binary_shape.in1_type = dtype;
  // Retarget computation type from bf16 to f32 due to missing hardware support.
  binary_shape.comp_type =
      dtype == LIBXSMM_DATATYPE_BF16 ? LIBXSMM_DATATYPE_F32 : dtype;
  binary_shape.out_type = dtype;
  binary_shape.ldi = static_cast<libxsmm_blasint>(ldiLhs);
  binary_shape.ldi2 = static_cast<libxsmm_blasint>(ldiRhs);
  binary_shape.ldo = static_cast<libxsmm_blasint>(ldo);

  libxsmm_meltwfunction_binary kernel = libxsmm_dispatch_meltw_binary_v2(
      static_cast<libxsmm_meltw_binary_type>(type), binary_shape,
      static_cast<libxsmm_bitfield>(binary_flags));
  if (!kernel) {
    fprintf(stderr, "failed to generate binary func\n");
    fprintf(stderr, "type: %ld\n", type);
    fprintf(stderr, "bcast_type: %ld\n", bcast_type);
    printXsmmStruct(binary_shape);
    exit(-1);
  }

  return reinterpret_cast<int64_t>(kernel);
}

extern "C" void
_mlir_ciface_xsmm_unary_invoke(const libxsmm_datatype dType, int64_t addr,
                               UnrankedMemRefType<char> *input,
                               UnrankedMemRefType<char> *output) {
  // std::cout << "tensor input: \n";
  // printMemRefMetaData(std::cout, DynamicMemRefType<char>(*input));
  // std::cout << "tensor output: \n";
  // printMemRefMetaData(std::cout, DynamicMemRefType<char>(*output));
  DynamicMemRefType<char> memrefA = DynamicMemRefType<char>(*input);
  DynamicMemRefType<char> memrefB = DynamicMemRefType<char>(*output);

  libxsmm_meltwfunction_unary kernel =
      reinterpret_cast<libxsmm_meltwfunction_unary>(addr);
  libxsmm_meltw_unary_param param;
  if (dType == LIBXSMM_DATATYPE_F32) {
    float *addr_a = (float *)memrefA.data + memrefA.offset;
    float *addr_b = (float *)memrefB.data + memrefB.offset;

    param.in.primary = (void *)addr_a;
    param.out.primary = (void *)addr_b;
  } else if (dType == LIBXSMM_DATATYPE_BF16) {

    bf16 *addr_a = (bf16 *)memrefA.data + memrefA.offset;
    bf16 *addr_b = (bf16 *)memrefB.data + memrefB.offset;

    param.in.primary = (void *)addr_a;
    param.out.primary = (void *)addr_b;
  }
  kernel(&param);
}

extern "C" void _mlir_ciface_xsmm_binary_invoke(const libxsmm_datatype dType,
                                                int64_t addr,
                                                UnrankedMemRefType<char> *lhs,
                                                UnrankedMemRefType<char> *rhs,
                                                UnrankedMemRefType<char> *out) {
  DynamicMemRefType<char> memrefLhs = DynamicMemRefType<char>(*lhs);
  DynamicMemRefType<char> memrefRhs = DynamicMemRefType<char>(*rhs);
  DynamicMemRefType<char> memrefOut = DynamicMemRefType<char>(*out);
  libxsmm_meltwfunction_binary kernel =
      reinterpret_cast<libxsmm_meltwfunction_binary>(addr);
  libxsmm_meltw_binary_param param;

  if (dType == LIBXSMM_DATATYPE_F32) {
    float *addr_memref_lhs = (float *)memrefLhs.data + memrefLhs.offset;
    float *addr_memref_rhs = (float *)memrefRhs.data + memrefRhs.offset;
    float *addr_memref_out = (float *)memrefOut.data + memrefOut.offset;
    param.in0.primary = (void *)addr_memref_lhs;
    param.in1.primary = (void *)addr_memref_rhs;
    param.out.primary = (void *)addr_memref_out;

  } else if (dType == LIBXSMM_DATATYPE_BF16) {
    bf16 *addr_memref_lhs = (bf16 *)memrefLhs.data + memrefLhs.offset;
    bf16 *addr_memref_rhs = (bf16 *)memrefRhs.data + memrefRhs.offset;
    bf16 *addr_memref_out = (bf16 *)memrefOut.data + memrefOut.offset;
    param.in0.primary = (void *)addr_memref_lhs;
    param.in1.primary = (void *)addr_memref_rhs;
    param.out.primary = (void *)addr_memref_out;
  }
  kernel(&param);
}

extern "C" void
_mlir_ciface_xsmm_unary_scalar_invoke(const libxsmm_datatype dType,
                                      int64_t addr, float input,
                                      UnrankedMemRefType<char> *output) {
  DynamicMemRefType<char> tensorB = DynamicMemRefType<char>(*output);
  libxsmm_meltwfunction_unary kernel =
      reinterpret_cast<libxsmm_meltwfunction_unary>(addr);
  libxsmm_meltw_unary_param param;

  param.in.primary = (void *)&input;
  param.out.primary = (void *)tensorB.data;
  kernel(&param);
}

extern "C" void _mlir_ciface_xsmm_brgemm_invoke(const libxsmm_datatype dType,
                                                int64_t addr,
                                                UnrankedMemRefType<char> *A,
                                                UnrankedMemRefType<char> *B,
                                                UnrankedMemRefType<char> *C,
                                                int64_t numBatches) {
  // std::cout << "numBatch: " << numBatches << "\n";
  // std::cout << "\n A: \n";
  // printMemRefMetaData(std::cout, DynamicMemRefType<char>(*A));
  // std::cout << "\n B: \n";
  // printMemRefMetaData(std::cout, DynamicMemRefType<char>(*B));
  // std::cout << "\n C: \n";
  // printMemRefMetaData(std::cout, DynamicMemRefType<char>(*C));
  // std::cout << "\n";

  DynamicMemRefType<char> tensorA = DynamicMemRefType<char>(*A);
  DynamicMemRefType<char> tensorB = DynamicMemRefType<char>(*B);
  DynamicMemRefType<char> tensorC = DynamicMemRefType<char>(*C);

  libxsmm_xmmfunction sgemm;
  libxsmm_gemm_param gemm_param;
  sgemm.gemm = reinterpret_cast<libxsmm_gemmfunction>(addr);
  unsigned long long numBatchesVar = numBatches;
  if (dType == LIBXSMM_DATATYPE_F32) {
    float *addr_tensorA = (float *)tensorA.data + tensorA.offset;
    float *addr_tensorB = (float *)tensorB.data + tensorB.offset;
    float *addr_tensorC = (float *)tensorC.data + tensorC.offset;
    gemm_param.a.primary = (void *)addr_tensorB;
    gemm_param.b.primary = (void *)addr_tensorA;
    gemm_param.c.primary = (void *)addr_tensorC;
  } else if (dType == LIBXSMM_DATATYPE_BF16) {
    bf16 *addr_tensorA = (bf16 *)tensorA.data + tensorA.offset;
    bf16 *addr_tensorB = (bf16 *)tensorB.data + tensorB.offset;
    bf16 *addr_tensorC = (bf16 *)tensorC.data + tensorC.offset;
    gemm_param.a.primary = (void *)addr_tensorB;
    gemm_param.b.primary = (void *)addr_tensorA;
    gemm_param.c.primary = (void *)addr_tensorC;
  }
  gemm_param.op.tertiary = (void *)&numBatchesVar;
  sgemm.gemm(&gemm_param);
}

extern "C" int64_t _mlir_ciface_xsmm_brgemm_dispatch(
    const libxsmm_datatype dtype, int64_t m, int64_t n, int64_t k, int64_t lda,
    int64_t ldb, int64_t ldc, const libxsmm_gemm_flags flags) {
  // std::cout << "lda: " << lda << "\n";
  // std::cout << "lbd: " << ldb << "\n";
  // std::cout << "ldc: " << ldc << "\n";
  // std::cout << "m: " << m << "\n";
  // std::cout << "n: " << n << "\n";
  // std::cout << "k: " << k << "\n";

  libxsmm_blasint lda_int = lda;
  libxsmm_blasint ldb_int = ldb;
  libxsmm_blasint ldc_int = ldc;
  libxsmm_blasint m_int = m;
  libxsmm_blasint n_int = n;
  libxsmm_blasint k_int = k;
  // TODO: move stride computation to dispatch
  // operation as in: https://github.com/plaidml/plaidml/pull/1983
  auto typeSize = dtype == LIBXSMM_DATATYPE_F32 ? sizeof(float) : sizeof(bf16);
  libxsmm_blasint stride_a = lda * m * typeSize;
  libxsmm_blasint stride_b = ldb * k * typeSize;

  libxsmm_gemm_shape l_shape;
  libxsmm_bitfield l_flags = LIBXSMM_GEMM_FLAG_NONE;
  if (flags == LIBXSMM_GEMM_FLAG_VNNI_B) {
    assert(dtype == LIBXSMM_DATATYPE_BF16);
    // We swap A and B since LIBXSMM is col-major.
    // We need to update the flag.
    l_flags = l_flags | LIBXSMM_GEMM_FLAG_VNNI_A;
  } else {
    l_flags = l_flags | flags;
  }
  libxsmm_bitfield l_prefetch_flags = 0;
  libxsmm_gemm_batch_reduce_config l_brconfig;

  l_shape.m = n_int;
  l_shape.n = m_int;
  l_shape.k = k_int;
  l_shape.lda = ldb_int;
  l_shape.ldb = lda_int;
  l_shape.ldc = ldc_int;
  l_shape.a_in_type = dtype;
  l_shape.b_in_type = dtype;
  l_shape.out_type = dtype;
  // Retarget computation type from bf16 to f32 due to missing hardware support.
  l_shape.comp_type =
      dtype == LIBXSMM_DATATYPE_BF16 ? LIBXSMM_DATATYPE_F32 : dtype;
  l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
  l_brconfig.br_stride_a_hint = stride_b;
  l_brconfig.br_stride_b_hint = stride_a;
  l_brconfig.br_unroll_hint = 0;

  auto sgemm = libxsmm_dispatch_brgemm_v2(l_shape, l_flags, l_prefetch_flags,
                                          l_brconfig);
  if (!sgemm) {
    fprintf(stderr, "failed to generate brgemm func\n");
    fprintf(stderr, "dtype: %u\n", dtype);
    printXsmmStruct(l_shape);
    printXsmmStruct(l_brconfig);
    exit(-1);
  }

  return reinterpret_cast<int64_t>(sgemm);
}

extern "C" void _mlir_ciface_xsmm_fused_brgemm_invoke(
    const libxsmm_datatype dType, int64_t addr, UnrankedMemRefType<char> *A,
    UnrankedMemRefType<char> *B, UnrankedMemRefType<char> *C,
    UnrankedMemRefType<char> *D, int64_t numBatches) {
  // std::cout << "numBatch: " << numBatches << "\n";
  // std::cout << "\n A: \n";
  // printMemRefMetaData(std::cout, DynamicMemRefType<char>(*A));
  // std::cout << "\n B: \n";
  // printMemRefMetaData(std::cout, DynamicMemRefType<char>(*B));
  // std::cout << "\n C: \n";
  // printMemRefMetaData(std::cout, DynamicMemRefType<char>(*C));
  // std::cout << "\n";

  DynamicMemRefType<char> tensorA = DynamicMemRefType<char>(*A);
  DynamicMemRefType<char> tensorB = DynamicMemRefType<char>(*B);
  DynamicMemRefType<char> tensorC = DynamicMemRefType<char>(*C);
  DynamicMemRefType<char> tensorD = DynamicMemRefType<char>(*D);

  libxsmm_xmmfunction sgemm;
  libxsmm_gemm_ext_param gemm_param;
  sgemm.gemm_ext = reinterpret_cast<libxsmm_gemmfunction_ext>(addr);

  unsigned long long numBatchesVar = numBatches;
  if (dType == LIBXSMM_DATATYPE_F32) {
    float *addr_tensorA = (float *)tensorA.data + tensorA.offset;
    float *addr_tensorB = (float *)tensorB.data + tensorB.offset;
    float *addr_tensorC = (float *)tensorC.data + tensorC.offset;
    float *addr_tensorD = (float *)tensorD.data + tensorD.offset;
    gemm_param.a.primary = (void *)addr_tensorB;
    gemm_param.b.primary = (void *)addr_tensorA;
    gemm_param.c.primary = (void *)addr_tensorC;
    gemm_param.d.primary = (void *)addr_tensorD;
  } else if (dType == LIBXSMM_DATATYPE_BF16) {
    bf16 *addr_tensorA = (bf16 *)tensorA.data + tensorA.offset;
    bf16 *addr_tensorB = (bf16 *)tensorB.data + tensorB.offset;
    bf16 *addr_tensorC = (bf16 *)tensorC.data + tensorC.offset;
    bf16 *addr_tensorD = (bf16 *)tensorD.data + tensorD.offset;
    gemm_param.a.primary = (void *)addr_tensorB;
    gemm_param.b.primary = (void *)addr_tensorA;
    gemm_param.c.primary = (void *)addr_tensorC;
    gemm_param.d.primary = (void *)addr_tensorD;
  }
  gemm_param.op.tertiary = (void *)&numBatchesVar;
  sgemm.gemm_ext(&gemm_param);
}

extern "C" int64_t _mlir_ciface_xsmm_fused_brgemm_dispatch(
    const libxsmm_datatype dtype, bool isVNNI, int64_t m, int64_t n, int64_t k,
    int64_t lda, int64_t ldb, int64_t ldc) {
  // std::cout << "lda: " << lda << "\n";
  // std::cout << "lbd: " << ldb << "\n";
  // std::cout << "ldc: " << ldc << "\n";
  // std::cout << "m: " << m << "\n";
  // std::cout << "n: " << n << "\n";
  // std::cout << "k: " << k << "\n";

  libxsmm_blasint lda_int = lda;
  libxsmm_blasint ldb_int = ldb;
  libxsmm_blasint ldc_int = ldc;
  libxsmm_blasint m_int = m;
  libxsmm_blasint n_int = n;
  libxsmm_blasint k_int = k;
  // TODO: move stride computation to dispatch
  // operation as in: https://github.com/plaidml/plaidml/pull/1983
  auto typeSize = dtype == LIBXSMM_DATATYPE_F32 ? sizeof(float) : sizeof(bf16);
  libxsmm_blasint stride_a = lda * m * typeSize;
  libxsmm_blasint stride_b = ldb * k * typeSize;

  libxsmm_gemm_shape l_shape;
  libxsmm_bitfield l_flags;
  if (isVNNI) {
    assert(dtype == LIBXSMM_DATATYPE_BF16);
    l_flags = LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N');
  } else {
    l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  }
  libxsmm_bitfield l_prefetch_flags = 0;

  l_shape.m = n_int;
  l_shape.n = m_int;
  l_shape.k = k_int;
  l_shape.lda = ldb_int;
  l_shape.ldb = lda_int;
  l_shape.ldc = ldc_int;
  l_shape.a_in_type = dtype;
  l_shape.b_in_type = dtype;
  l_shape.out_type = dtype;
  // Retarget computation type from bf16 to f32 due to missing hardware support.
  l_shape.comp_type =
      dtype == LIBXSMM_DATATYPE_BF16 ? LIBXSMM_DATATYPE_F32 : dtype;

  libxsmm_gemm_batch_reduce_config l_brconfig;
  l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
  l_brconfig.br_stride_a_hint = stride_b;
  l_brconfig.br_stride_b_hint = stride_a;
  l_brconfig.br_unroll_hint = 0;

  libxsmm_gemm_ext_unary_argops l_argops;
  memset(&l_argops, 0, sizeof(libxsmm_gemm_ext_unary_argops));
  l_argops.ldcp = ldc;
  l_argops.cp_unary_type = LIBXSMM_MELTW_TYPE_UNARY_RELU;

  libxsmm_gemm_ext_binary_postops l_postops;
  memset(&l_postops, 0, sizeof(libxsmm_gemm_ext_binary_postops));
  l_postops.d_in_type = dtype;

  l_postops.d_binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0;
  l_postops.d_binary_type = LIBXSMM_MELTW_TYPE_BINARY_ADD;
  l_postops.ldd = ldc;

  auto sgemm = libxsmm_dispatch_brgemm_ext_v2(
      l_shape, l_flags, l_prefetch_flags, l_brconfig, l_argops, l_postops);
  if (!sgemm) {
    fprintf(stderr, "failed to generate fused brgemm func\n");
    fprintf(stderr, "dtype: %u\n", dtype);
    printXsmmStruct(l_shape);
    printXsmmStruct(l_brconfig);
    exit(-1);
  }

  return reinterpret_cast<int64_t>(sgemm);
}
//----------------------------------------------------------------------------//
// BRGEMM connection on the IREE side.
//----------------------------------------------------------------------------//

namespace {
// Definition of this struct needs to match with the definition used in
// MemrefToLLVM pass.
// https://mlir.llvm.org/doxygen/TypeConverter_8cpp_source.html#l00283
//
// This definition is used by LLVM to convert Memref into C struct in order to
// pass to our iree_*_invoke functions.
typedef struct {
  void *allocatedPtr;
  void *alignedPtr;
  int64_t offset;
  int64_t sizes_and_strides[0]; // variable size: sizes[rank], strides[rank]
} iree_input_tensor_t;

// Based on the code
// https://mlir.llvm.org/doxygen/TypeConverter_8cpp_source.html#l00283
// the offset needs to be 0. We add this assumption as assert.
// We will remove it if we find a violation of this assumption.
//
// We do not enforce the constraint that allocatedPtr and alignedPtr are same.
// Because allocatedPtr could be deadbeef in certain cases such as
// mlir/lib/Conversion/MemRefToLLVM/MemRefToLLVM.cpp#L739
static inline void
check_integrity_of_iree_input_tensor(const iree_input_tensor_t *tensor) {
  assert(tensor->offset == 0 &&
         "offset is non-zero in iree_input_tensor to XsmmRunner");
}
} // namespace

extern "C" int iree_xsmm_brgemm_dispatch(void *context, void *params,
                                         void *reserved) {
  typedef struct {
    int64_t address;
    int64_t dtype;
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t lda;
    int64_t ldb;
    int64_t ldc;
    const libxsmm_gemm_flags flags;
  } xsmm_brgemm_dispatch_t;
  xsmm_brgemm_dispatch_t *p = (xsmm_brgemm_dispatch_t *) params;
  p->address =
      _mlir_ciface_xsmm_brgemm_dispatch((libxsmm_datatype)p->dtype, p->m, p->n,
                                        p->k, p->lda, p->ldb, p->ldc, p->flags);
  return 0;
}

extern "C" int iree_xsmm_matmul_dispatch(void *context, void *params,
                                         void *reserved) {
  typedef struct {
    int64_t gemm_addr;
    int64_t dtype;
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t lda;
    int64_t ldb;
    int64_t ldc;
    const libxsmm_gemm_flags flags;
  } xsmm_matmul_dispatch_t;
  xsmm_matmul_dispatch_t *p = (xsmm_matmul_dispatch_t *) params;
  p->gemm_addr =
      _mlir_ciface_xsmm_matmul_dispatch((libxsmm_datatype)p->dtype, p->m, p->n,
                                        p->k, p->lda, p->ldb, p->ldc, p->flags);
  return 0;
}

extern "C" int iree_xsmm_unary_dispatch(void *context, void *params,
                                        void *reserved) {
  typedef struct {
    int64_t address;
    int64_t dtype;
    int64_t m;
    int64_t n;
    int64_t ldi;
    int64_t ldo;
    int64_t type;
    int64_t bcast_type;
  } xsmm_unary_dispatch;
  xsmm_unary_dispatch *p = (xsmm_unary_dispatch *) params;
  p->address =
      _mlir_ciface_xsmm_unary_dispatch((libxsmm_datatype) p->dtype, p->m, p->n,
                                       p->ldi, p->ldo, p->type, p->bcast_type);
  return 0;
}

extern "C" int iree_xsmm_binary_dispatch(void *context, void *params,
                                         void *reserved) {
  typedef struct {
    int64_t address;
    int64_t dtype;
    int64_t m;
    int64_t n;
    int64_t ldiLhs;
    int64_t ldiRhs;
    int64_t ldo;
    int64_t type;
    int64_t bcast_type;
  } xsmm_binary_dispatch;
  xsmm_binary_dispatch *p = (xsmm_binary_dispatch *) params;
  p->address = _mlir_ciface_xsmm_binary_dispatch(
      (libxsmm_datatype) p->dtype, p->m, p->n, p->ldiLhs, p->ldiRhs, p->ldo,
      p->type, p->bcast_type);
  return 0;
}

// TODO: struct slicing. BRGEMM struct is the same as the GEMM one plus the
// batch parameter.
extern "C" int iree_xsmm_brgemm_invoke(void *context, void *params,
                                       void *reserved) {
  typedef struct {
    int64_t dtype;
    int64_t address;
    int64_t rankA;
    iree_input_tensor_t *pA;
    int64_t rankB;
    iree_input_tensor_t *pB;
    int64_t rankC;
    iree_input_tensor_t *pC;
    int64_t numBatches;
  } xsmm_brgemm_invoke_t;
  xsmm_brgemm_invoke_t *p = (xsmm_brgemm_invoke_t *) params;

  check_integrity_of_iree_input_tensor(p->pA);
  check_integrity_of_iree_input_tensor(p->pB);
  check_integrity_of_iree_input_tensor(p->pC);

  void *addr_tensorA = p->pA->alignedPtr;
  void *addr_tensorB = p->pB->alignedPtr;
  void *addr_tensorC = p->pC->alignedPtr;

  libxsmm_xmmfunction sgemm;
  libxsmm_gemm_param gemm_param;
  sgemm.gemm = reinterpret_cast<libxsmm_gemmfunction>(p->address);
  unsigned long long numBatchesVar = p->numBatches;
  // LIBXSMM col-major change A with B.
  gemm_param.a.primary = (void *) addr_tensorB;
  gemm_param.b.primary = (void *) addr_tensorA;
  gemm_param.c.primary = (void *) addr_tensorC;
  gemm_param.op.tertiary = (void *) &numBatchesVar;
  sgemm.gemm(&gemm_param);

  return 0;
}

extern "C" int iree_xsmm_matmul_invoke(void *context, void *params,
                                       void *reserved) {
  typedef struct {
    int64_t dtype;
    int64_t gemm_addr;
    int64_t rankA;
    iree_input_tensor_t *pA;
    int64_t rankB;
    iree_input_tensor_t *pB;
    int64_t rankC;
    iree_input_tensor_t *pC;
  } xsmm_matmul_invoke_t;
  xsmm_matmul_invoke_t *p = (xsmm_matmul_invoke_t *) params;

  check_integrity_of_iree_input_tensor(p->pA);
  check_integrity_of_iree_input_tensor(p->pB);
  check_integrity_of_iree_input_tensor(p->pC);

  void *addr_tensorA = p->pA->alignedPtr;
  void *addr_tensorB = p->pB->alignedPtr;
  void *addr_tensorC = p->pC->alignedPtr;

  libxsmm_xmmfunction sgemm;
  libxsmm_gemm_param gemm_param;
  // LIBXSMM col-major change A with B.
  gemm_param.a.primary = (void *) addr_tensorB;
  gemm_param.b.primary = (void *) addr_tensorA;
  gemm_param.c.primary = (void *) addr_tensorC;
  sgemm.gemm = reinterpret_cast<libxsmm_gemmfunction>(p->gemm_addr);

  sgemm.gemm(&gemm_param);
  return 0;
}

extern "C" int iree_xsmm_unary_invoke(void *context, void *params,
                                      void *reserved) {
  typedef struct {
    int64_t dtype;
    int64_t address;
    int64_t rankA;
    iree_input_tensor_t *pA;
    int64_t rankB;
    iree_input_tensor_t *pB;
  } xsmm_unary_invoke;
  xsmm_unary_invoke *p = (xsmm_unary_invoke *) params;

  check_integrity_of_iree_input_tensor(p->pA);
  check_integrity_of_iree_input_tensor(p->pB);

  void *addr_a = p->pA->alignedPtr;
  void *addr_b = p->pB->alignedPtr;

  libxsmm_meltwfunction_unary kernel =
      reinterpret_cast<libxsmm_meltwfunction_unary>(p->address);
  libxsmm_meltw_unary_param param;
  param.in.primary = (void *) addr_a;
  param.out.primary = (void *) addr_b;
  kernel(&param);

  return 0;
}

extern "C" int iree_xsmm_binary_invoke(void *context, void *params,
                                       void *reserved) {
  typedef struct {
    int64_t dtype;
    int64_t address;
    int64_t rankA;
    iree_input_tensor_t *pA;
    int64_t rankB;
    iree_input_tensor_t *pB;
    int64_t rankC;
    iree_input_tensor_t *pC;
  } xsmm_binary_invoke;
  xsmm_binary_invoke *p = (xsmm_binary_invoke *) params;

  check_integrity_of_iree_input_tensor(p->pA);
  check_integrity_of_iree_input_tensor(p->pB);
  check_integrity_of_iree_input_tensor(p->pC);

  void *addr_a = p->pA->alignedPtr;
  void *addr_b = p->pB->alignedPtr;
  void *addr_c = p->pC->alignedPtr;

  libxsmm_meltwfunction_binary kernel =
      reinterpret_cast<libxsmm_meltwfunction_binary>(p->address);
  libxsmm_meltw_binary_param param;
  param.in0.primary = (void *) addr_a;
  param.in1.primary = (void *) addr_b;
  param.out.primary = (void *) addr_c;
  kernel(&param);

  return 0;
}

static void printXsmmStruct(const libxsmm_gemm_shape &gemmShape,
                            FILE *outfile) {
  fprintf(outfile, "M: %d\n", gemmShape.m);
  fprintf(outfile, "N: %d\n", gemmShape.n);
  fprintf(outfile, "K: %d\n", gemmShape.k);
  fprintf(outfile, "lda: %d\n", gemmShape.lda);
  fprintf(outfile, "ldb: %d\n", gemmShape.ldb);
  fprintf(outfile, "ldc: %d\n", gemmShape.ldc);
  fprintf(outfile, "a_in_type: %d\n", gemmShape.a_in_type);
  fprintf(outfile, "b_in_type: %d\n", gemmShape.b_in_type);
  fprintf(outfile, "comp_type: %d\n", gemmShape.comp_type);
  fprintf(outfile, "out_type: %d\n", gemmShape.out_type);
}

static void printXsmmStruct(const libxsmm_meltw_unary_shape &unaryShape,
                            FILE *outfile) {
  fprintf(outfile, "M: %d\n", unaryShape.m);
  fprintf(outfile, "N: %d\n", unaryShape.n);
  fprintf(outfile, "in0_type: %d\n", unaryShape.in0_type);
  fprintf(outfile, "comp_type: %d\n", unaryShape.comp_type);
  fprintf(outfile, "out_type: %d\n", unaryShape.out_type);
  fprintf(outfile, "ldi: %d\n", unaryShape.ldi);
  fprintf(outfile, "ldo: %d\n", unaryShape.ldo);
}

static void printXsmmStruct(const libxsmm_meltw_binary_shape &binaryShape,
                            FILE *outfile) {
  fprintf(outfile, "M: %d\n", binaryShape.m);
  fprintf(outfile, "N: %d\n", binaryShape.n);
  fprintf(outfile, "in0_type: %d\n", binaryShape.in0_type);
  fprintf(outfile, "in1_type: %d\n", binaryShape.in1_type);
  fprintf(outfile, "comp_type: %d\n", binaryShape.comp_type);
  fprintf(outfile, "out_type: %d\n", binaryShape.out_type);
  fprintf(outfile, "ldi: %d\n", binaryShape.ldi);
  fprintf(outfile, "ldi2: %d\n", binaryShape.ldi2);
  fprintf(outfile, "ldo: %d\n", binaryShape.ldo);
}

static void
printXsmmStruct(const libxsmm_gemm_batch_reduce_config &brgemmConfig,
                FILE *outfile) {
  fprintf(outfile, "br_type: %d\n", brgemmConfig.br_type);
  fprintf(outfile, "br_stride_a_hint: %d\n", brgemmConfig.br_stride_a_hint);
  fprintf(outfile, "br_stride_b_hint: %d\n", brgemmConfig.br_stride_b_hint);
  fprintf(outfile, "br_unroll_hint: %d\n", brgemmConfig.br_unroll_hint);
}
