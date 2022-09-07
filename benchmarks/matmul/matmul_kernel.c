#include "../memref.h"
#include <libxsmm.h>

#if !defined(NO_JIT) && ((0 == LIBXSMM_JIT) || 0)
#define NO_JIT
#endif

/* Kernel implementation not based on MLIR (stub) */
void matmul(DECL_VEC2D_FUNC_IN_ARGS(a, float),
            DECL_VEC2D_FUNC_IN_ARGS(b, float),
            DECL_VEC2D_FUNC_OUT_ARGS(c, float)) {
  const libxsmm_blasint lda = (libxsmm_blasint)b_stride0;
  const libxsmm_blasint ldb = (libxsmm_blasint)a_stride0;
  const libxsmm_blasint ldc = (libxsmm_blasint)c_stride0;
  const libxsmm_blasint m = (libxsmm_blasint)c_size1;
  const libxsmm_blasint n = (libxsmm_blasint)c_size0;
  const libxsmm_blasint k = (libxsmm_blasint)a_size1;
#if defined(NO_JIT)
  const float alpha = 1.f, beta = 1.f;
  LIBXSMM_INLINE_XGEMM(float, float, "N", "N", &m, &n, &k, &alpha, b_alignedPtr,
                       &lda, a_alignedPtr, &ldb, &beta, c_alignedPtr, &ldc);
#else
  const libxsmm_gemm_shape gemm_shape = libxsmm_create_gemm_shape(
      m, n, k, lda, ldb, ldc, LIBXSMM_DATATYPE(float), LIBXSMM_DATATYPE(float),
      LIBXSMM_DATATYPE(float), LIBXSMM_DATATYPE(float));
  libxsmm_xmmfunction kernel = {NULL};
  kernel.gemm = libxsmm_dispatch_gemm_v2(gemm_shape, LIBXSMM_GEMM_FLAG_NONE,
                                         LIBXSMM_PREFETCH_NONE);
  libxsmm_gemm_param gemm_param;
  // memset(&gemm_param, 0, sizeof(gemm_param));
  gemm_param.a.primary = (void *)b_alignedPtr;
  gemm_param.b.primary = (void *)a_alignedPtr;
  gemm_param.c.primary = c_alignedPtr;
  // no prefetch (gemm_param.?.quaternary not used)
  kernel.gemm(&gemm_param);
#endif
}
