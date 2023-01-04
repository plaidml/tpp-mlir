#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "../utils.h"
#include "../Perf.h"

//#include <libxsmm.h>

#if !defined(NO_JIT) && ((0 == LIBXSMM_JIT) || 0)
#define NO_JIT
#endif

/*
void matmul_xsmm(DECL_VEC2D_FUNC_IN_ARGS(a, float),
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
*/

void matmul_xsmm(const float *A, const float* B, float* O, int m, int n, int k) {
  assert(0 && "Not implemented yet");
}

int main(int argc, char *argv[]) {
  float* A;
  float* B;
  float* O;
  int verbose = 0;
  void(*kernel)(const float*,const float*,float*,int,int,int) = matmul_ref;

  int m = 0;
  int n = 0;
  int k = 0;
  int iter = 10;

  // Cmd-line args
  int opt;
  while ((opt = getopt(argc, argv, "n:xv")) != -1) {
    switch (opt) {
      case 'n':
        iter = atoi(optarg);
        break;
      case 'x':
        kernel = matmul_xsmm;
        break;
      case 'v':
        verbose++;
        break;
      default:
        fprintf(stderr, "Invalid argument %c %s\n", opt, optarg);
        fprintf(stderr, "Syntax: %s [-v] [-n NN]\n", argv[0]);
        exit(1);
    }
  }
  // parse MxNxK as last argument
  if (sscanf(argv[argc-1], "%dx%dx%d", &m, &n, &k) != 3) {
    fprintf(stderr, "Dims argument must be format MxNxK, not '%s'\n", argv[argc-1]);
    exit(1);
  }
  double gflops = (double)(2*n*m*k) / 1e9;
  if (verbose) {
    fprintf(stderr, "Kernel version: %s\n", (kernel == matmul_ref) ? "ref" : "xsmm");
    fprintf(stderr, "[ %d, %d ] = [ %d, %d ] x [ %d, %d ] (%dx)\n", m, n, m, k, k, n, iter);
    fprintf(stderr, "%lf Gfps\n", gflops);
  }

  // Initialize input
  A = (float*)malloc(m*k*sizeof(float));
  B = (float*)malloc(k*n*sizeof(float));
  O = (float*)malloc(m*n*sizeof(float));
  if (!A || !B || !O) {
    fprintf(stderr, "Allocation failed\n");
    exit(1);
  }
  init_matrix(A, m, k);
  init_matrix(B, k, n);
  clear_matrix(O, m, n);
  if (verbose > 2) {
    dump_matrix(A, m, k, "A");
    dump_matrix(B, k, n, "B");
    dump_matrix(O, m, n, "O");
    puts("");
  }

  // Warmup
  if (verbose > 2)
    puts("MATMUL\n");

  kernel(A, B, O, m, n, k);

  if (verbose > 2)
    dump_matrix(O, m, n, "O");

  // Run the reference benchmark
  if (verbose > 2)
    puts("BENCH");

  PerfResults perf;
  for (int i=0; i<iter; i++) {
    perf.startTimer();
    kernel(A, B, O, m, n, k);
    perf.stopTimer();
  }

  double mean = perf.getMean();
  double stdev = perf.getStdev();

  if (verbose)
    fprintf(stderr, "%3.9f +- %3.9f ms\n", mean, stdev);
  printf("%9.3f +- %9.3f gflops\n", gflops/mean, (gflops*stdev)/(mean*mean));

  free(A);
  free(B);
  free(O);

  return 0;
}
