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

// Helper function prototypes.
static void printXsmmStruct(const libxsmm_gemm_shape &gemmShape,
                            FILE *outfile = stderr);
static void printXsmmStruct(const libxsmm_meltw_unary_shape &unaryShape,
                            FILE *outfile = stderr);
static void printXsmmStruct(const libxsmm_meltw_binary_shape &binaryShape,
                            FILE *outfile = stderr);
static void printXsmmStruct(const libxsmm_gemm_batch_reduce_config &brgemmShape,
                            FILE *outfile = stderr);

static bool hasImplicitComputeDtypeUnary(const libxsmm_meltw_unary_type dtype) {
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
} mlir_memref_descriptor_t;

void *get_data_pointer_from_memref_desc(const libxsmm_datatype dType,
                                        void *memrefDesc) {
  mlir_memref_descriptor_t *desc = (mlir_memref_descriptor_t *)memrefDesc;
  if (dType == LIBXSMM_DATATYPE_F32) {
    float *data_pointer = (float *)desc->alignedPtr + desc->offset;
    return (void *)data_pointer;
  } else if (dType == LIBXSMM_DATATYPE_BF16) {
    bf16 *data_pointer = (bf16 *)desc->alignedPtr + desc->offset;
    return (void *)data_pointer;
  }
  fprintf(stderr, "Unhandled data type in get_data_pointer_from_memref_desc:%d",
          dType);
  return nullptr;
}

} // namespace

extern "C" void xsmm_gemm_invoke(const libxsmm_datatype dType, int64_t addr,
                                 int64_t rankA, void *memrefDescA,
                                 int64_t rankB, void *memrefDescB,
                                 int64_t rankC, void *memrefDescC) {
  libxsmm_xmmfunction sgemm;
  libxsmm_gemm_param gemm_param;

  // LIBXSMM col-major change A with B.
  gemm_param.a.primary = get_data_pointer_from_memref_desc(dType, memrefDescB);
  gemm_param.b.primary = get_data_pointer_from_memref_desc(dType, memrefDescA);
  gemm_param.c.primary = get_data_pointer_from_memref_desc(dType, memrefDescC);

  sgemm.gemm = reinterpret_cast<libxsmm_gemmfunction>(addr);
  sgemm.gemm(&gemm_param);
}

extern "C" int64_t xsmm_gemm_dispatch(const libxsmm_datatype dtype, int64_t m,
                                      int64_t n, int64_t k, int64_t lda,
                                      int64_t ldb, int64_t ldc,
                                      const libxsmm_gemm_flags flags) {
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
  libxsmm_bitfield l_flags = flags;
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
xsmm_unary_dispatch(const libxsmm_meltw_unary_type op_type,
                    const libxsmm_datatype dtype, int64_t m, int64_t n,
                    int64_t ldi, int64_t ldo,
                    const libxsmm_meltw_unary_flags unary_flags) {
  // std::cout << "ldi: " << ldi << "\n";
  // std::cout << "ldo: " << ldo << "\n";
  // std::cout << "m: " << m << "\n";
  // std::cout << "n: " << n << "\n";
  // std::cout << "type: " << type << "\n";
  // std::cout << "bcast_type: " << bcast_type << "\n";

  libxsmm_meltw_unary_shape unary_shape;
  // Row major to col major swap m with n.
  unary_shape.m = static_cast<libxsmm_blasint>(n);
  unary_shape.n = static_cast<libxsmm_blasint>(m);
  unary_shape.in0_type = dtype;
  // Retarget computation type from bf16 to f32 due to missing hardware support.
  // Copy and Zero should remain in BF16 to avoid useless up/down casts
  auto force_fp32 =
      (dtype == LIBXSMM_DATATYPE_BF16 && !hasImplicitComputeDtypeUnary(op_type));
  unary_shape.comp_type = force_fp32 ? LIBXSMM_DATATYPE_F32 : dtype;
  unary_shape.out_type = dtype;
  unary_shape.ldi = static_cast<libxsmm_blasint>(ldi);
  unary_shape.ldo = static_cast<libxsmm_blasint>(ldo);

  libxsmm_meltwfunction_unary kernel =
      libxsmm_dispatch_meltw_unary_v2(op_type, unary_shape, unary_flags);
  if (!kernel) {
    fprintf(stderr, "failed to generate unary func\n");
    fprintf(stderr, "op_type: %u\n", op_type);
    fprintf(stderr, "flags: %u\n", unary_flags);
    printXsmmStruct(unary_shape);
    exit(-1);
  }

  return reinterpret_cast<int64_t>(kernel);
}

extern "C" int64_t
xsmm_binary_dispatch(const libxsmm_meltw_binary_type op_type,
                     const libxsmm_datatype dtype, int64_t m, int64_t n,
                     int64_t ldiLhs, int64_t ldiRhs, int64_t ldo,
                     const libxsmm_meltw_binary_flags flags) {
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

  libxsmm_meltwfunction_binary kernel =
      libxsmm_dispatch_meltw_binary_v2(op_type, binary_shape, flags);
  if (!kernel) {
    fprintf(stderr, "failed to generate binary func\n");
    fprintf(stderr, "op_type: %u\n", op_type);
    fprintf(stderr, "flags: %u\n", flags);
    printXsmmStruct(binary_shape);
    exit(-1);
  }

  return reinterpret_cast<int64_t>(kernel);
}

extern "C" void xsmm_unary_invoke(const libxsmm_datatype dType, int64_t addr,
                                  int64_t inputRank, void *inputMemrefDesc,
                                  int64_t outputRank, void *outputMemrefDesc) {
  libxsmm_meltw_unary_param param;

  param.in.primary = get_data_pointer_from_memref_desc(dType, inputMemrefDesc);
  param.out.primary =
      get_data_pointer_from_memref_desc(dType, outputMemrefDesc);

  libxsmm_meltwfunction_unary kernel =
      reinterpret_cast<libxsmm_meltwfunction_unary>(addr);
  kernel(&param);
}

extern "C" void xsmm_binary_invoke(const libxsmm_datatype dType, int64_t addr,
                                   int64_t lhsRank, void *lhsMemrefDesc,
                                   int64_t rhsRank, void *rhsMemrefDesc,
                                   int64_t outRank, void *outMemrefDesc) {
  libxsmm_meltw_binary_param param;

  param.in0.primary = get_data_pointer_from_memref_desc(dType, lhsMemrefDesc);
  param.in1.primary = get_data_pointer_from_memref_desc(dType, rhsMemrefDesc);
  param.out.primary = get_data_pointer_from_memref_desc(dType, outMemrefDesc);

  libxsmm_meltwfunction_binary kernel =
      reinterpret_cast<libxsmm_meltwfunction_binary>(addr);
  kernel(&param);
}

extern "C" void xsmm_unary_scalar_invoke(const libxsmm_datatype dType,
                                         int64_t addr, float input,
                                         int64_t rankOutput,
                                         void *outputMemrefDesc) {
  libxsmm_meltw_unary_param param;

  param.in.primary = (void *)&input;
  param.out.primary =
      get_data_pointer_from_memref_desc(dType, outputMemrefDesc);

  libxsmm_meltwfunction_unary kernel =
      reinterpret_cast<libxsmm_meltwfunction_unary>(addr);
  kernel(&param);
}

extern "C" void xsmm_brgemm_invoke(const libxsmm_datatype dType, int64_t addr,
                                   int64_t rankA, void *memrefDescA,
                                   int64_t rankB, void *memrefDescB,
                                   int64_t rankC, void *memrefDescC,
                                   int64_t numBatches) {
  libxsmm_xmmfunction sgemm;
  libxsmm_gemm_param gemm_param;

  unsigned long long numBatchesVar = numBatches;
  gemm_param.op.tertiary = (void *)&numBatchesVar;

  // LIBXSMM col-major change A with B.
  gemm_param.a.primary = get_data_pointer_from_memref_desc(dType, memrefDescB);
  gemm_param.b.primary = get_data_pointer_from_memref_desc(dType, memrefDescA);
  gemm_param.c.primary = get_data_pointer_from_memref_desc(dType, memrefDescC);

  sgemm.gemm = reinterpret_cast<libxsmm_gemmfunction>(addr);
  sgemm.gemm(&gemm_param);
}

extern "C" int64_t xsmm_brgemm_dispatch(const libxsmm_datatype dtype, int64_t m,
                                        int64_t n, int64_t k, int64_t lda,
                                        int64_t ldb, int64_t ldc,
                                        const libxsmm_gemm_flags flags) {
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
  libxsmm_bitfield l_flags = flags;
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

extern "C" void
xsmm_fused_brgemm_invoke(const libxsmm_datatype dType, int64_t addr,
                         int64_t rankA, void *memrefDescA, int64_t rankB,
                         void *memrefDescB, int64_t rankC, void *memrefDescC,
                         int64_t rankD, void *memrefDescD, int64_t numBatches) {
  libxsmm_xmmfunction sgemm;
  libxsmm_gemm_ext_param gemm_param;

  unsigned long long numBatchesVar = numBatches;
  gemm_param.op.tertiary = (void *)&numBatchesVar;

  // LIBXSMM col-major change A with B.
  gemm_param.a.primary = get_data_pointer_from_memref_desc(dType, memrefDescB);
  gemm_param.b.primary = get_data_pointer_from_memref_desc(dType, memrefDescA);
  gemm_param.c.primary = get_data_pointer_from_memref_desc(dType, memrefDescC);
  gemm_param.d.primary = get_data_pointer_from_memref_desc(dType, memrefDescD);

  sgemm.gemm_ext = reinterpret_cast<libxsmm_gemmfunction_ext>(addr);
  sgemm.gemm_ext(&gemm_param);
}

extern "C" int64_t xsmm_fused_brgemm_dispatch(const libxsmm_datatype dtype,
                                              bool isVNNI, int64_t m, int64_t n,
                                              int64_t k, int64_t lda,
                                              int64_t ldb, int64_t ldc) {
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
  p->address = xsmm_brgemm_dispatch((libxsmm_datatype)p->dtype, p->m, p->n,
                                    p->k, p->lda, p->ldb, p->ldc, p->flags);
  return 0;
}

extern "C" int iree_xsmm_gemm_dispatch(void *context, void *params,
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
  } xsmm_gemm_dispatch_t;
  xsmm_gemm_dispatch_t *p = (xsmm_gemm_dispatch_t *)params;
  p->gemm_addr = xsmm_gemm_dispatch((libxsmm_datatype)p->dtype, p->m, p->n,
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
    const libxsmm_meltw_unary_flags flags;
  } xsmm_unary_dispatch_t;
  xsmm_unary_dispatch_t *p = (xsmm_unary_dispatch_t *)params;
  p->address = xsmm_unary_dispatch(
      (libxsmm_meltw_unary_type)p->type, (libxsmm_datatype)p->dtype, p->m, p->n,
      p->ldi, p->ldo, (libxsmm_meltw_unary_flags)p->flags);
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
    const libxsmm_meltw_binary_flags flags;
  } xsmm_binary_dispatch_t;
  xsmm_binary_dispatch_t *p = (xsmm_binary_dispatch_t *)params;
  p->address = xsmm_binary_dispatch(
      (libxsmm_meltw_binary_type)p->type, (libxsmm_datatype)p->dtype, p->m,
      p->n, p->ldiLhs, p->ldiRhs, p->ldo, (libxsmm_meltw_binary_flags)p->flags);
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
    mlir_memref_descriptor_t *memrefDescA;
    int64_t rankB;
    mlir_memref_descriptor_t *memrefDescB;
    int64_t rankC;
    mlir_memref_descriptor_t *memrefDescC;
    int64_t numBatches;
  } xsmm_brgemm_invoke_t;
  xsmm_brgemm_invoke_t *p = (xsmm_brgemm_invoke_t *) params;

  xsmm_brgemm_invoke((libxsmm_datatype)p->dtype, p->address, p->rankA,
                     p->memrefDescA, p->rankB, p->memrefDescB, p->rankC,
                     p->memrefDescC, p->numBatches);
  return 0;
}

extern "C" int iree_xsmm_gemm_invoke(void *context, void *params,
                                     void *reserved) {
  typedef struct {
    int64_t dtype;
    int64_t gemm_addr;
    int64_t rankA;
    mlir_memref_descriptor_t *memrefDescA;
    int64_t rankB;
    mlir_memref_descriptor_t *memrefDescB;
    int64_t rankC;
    mlir_memref_descriptor_t *memrefDescC;
  } xsmm_gemm_invoke_t;
  xsmm_gemm_invoke_t *p = (xsmm_gemm_invoke_t *)params;
  xsmm_gemm_invoke((libxsmm_datatype)p->dtype, p->gemm_addr, p->rankA,
                   p->memrefDescA, p->rankB, p->memrefDescB, p->rankC,
                   p->memrefDescC);
  return 0;
}

extern "C" int iree_xsmm_unary_invoke(void *context, void *params,
                                      void *reserved) {
  typedef struct {
    int64_t dtype;
    int64_t address;
    int64_t inputRank;
    mlir_memref_descriptor_t *inputMemrefDesc;
    int64_t outputRank;
    mlir_memref_descriptor_t *outputMemrefDesc;
  } xsmm_unary_invoke_t;
  xsmm_unary_invoke_t *p = (xsmm_unary_invoke_t *)params;

  xsmm_unary_invoke((libxsmm_datatype)p->dtype, p->address, p->inputRank,
                    p->inputMemrefDesc, p->outputRank, p->outputMemrefDesc);
  return 0;
}

extern "C" int iree_xsmm_binary_invoke(void *context, void *params,
                                       void *reserved) {
  typedef struct {
    int64_t dtype;
    int64_t address;
    int64_t lhsRank;
    mlir_memref_descriptor_t *lhsMemrefDesc;
    int64_t rhsRank;
    mlir_memref_descriptor_t *rhsMemrefDesc;
    int64_t outRank;
    mlir_memref_descriptor_t *outMemrefDesc;
  } xsmm_binary_invoke_t;
  xsmm_binary_invoke_t *p = (xsmm_binary_invoke_t *)params;

  xsmm_binary_invoke((libxsmm_datatype)p->dtype, p->address, p->lhsRank,
                     p->lhsMemrefDesc, p->rhsRank, p->rhsMemrefDesc, p->outRank,
                     p->outMemrefDesc);
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
