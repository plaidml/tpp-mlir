#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "../utils.h"

/* Reference implementation of a mlp */
// A = 128 x 256
// B = 256 x 512
// C =   1 x 512
// O = 128 x 512
// Total flops = broadcast O(n*m) + matmul O(n*m*k) + ReLU (O(n*m)
// 128x512x2 (131072) + 128x256x512 (16777216) = 16908288

void mlp_ref(const float *A, const float* B, const float* C, float* O, int m, int n, int k) {
  // BROADCAST INIT O from C
  for (int mi = 0; mi < m; ++mi) {
    for (int ni = 0; ni < n; ++ni) {
      O[mi*n+ni] = C[ni];
    }
  }

  // MATMUL O += A x B
  matmul_ref(A, B, O, m, n, k);

  // RELU
  for (int ni = 0; ni < n; ++ni) {
    for (int mi = 0; mi < m; mi++) {
      O[mi*n+ni] = MAX(0, O[mi*n+ni]);
    }
  }
}

void mlp_xsmm(const float *A, const float* B, const float* C, float* O, int m, int n, int k) {
  assert(0 && "Not implemented yet");
}

int main(int argc, char *argv[]) {
  float* A;
  float* B;
  float* C;
  float* O;
  int verbose = 0;
  void(*kernel)(const float*,const float*,const float*,float*,int,int,int) = mlp_ref;

  int m = 128;
  int n = 512;
  int k = 256;
  int iter = 10;

  // Cmd-line args
  int opt;
  while ((opt = getopt(argc, argv, "i:xv")) != -1) {
    switch (opt) {
      case 'i':
        iter = atoi(optarg);
        break;
      case 'x':
        kernel = mlp_xsmm;
        break;
      case 'v':
        verbose++;
        break;
      default:
        printf("Invalid argument %c %s\n", opt, optarg);
        printf("Syntax: %s [-v] [-n NN]\n", argv[0]);
        exit(1);
    }
  }
  double gflops = (double)((n*m) + (2*n*m*k) + (n*m)) / 1e9;
  if (verbose) {
    printf("Kernel version: %s\n", (kernel == mlp_ref) ? "ref" : "xsmm");
    printf("[ %d, %d ] = [ %d, %d ] x [ %d, %d ] (%dx)\n", m, n, m, k, k, n, iter);
    printf("%lf Gfps\n", gflops);
  }

  // Initialize input
  A = (float*)malloc(m*k*sizeof(float));
  B = (float*)malloc(k*n*sizeof(float));
  C = (float*)malloc(n*sizeof(float));
  O = (float*)malloc(m*n*sizeof(float));
  if (!A || !B || !C || !O) {
    fprintf(stderr, "Allocation failed");
    exit(1);
  }
  init_matrix(A, m, k);
  init_matrix(B, k, n);
  init_matrix(C, 1, n);
  clear_matrix(O, m, n);
  if (verbose > 1) {
    dump_matrix(A, m, k, "A");
    dump_matrix(B, k, n, "B");
    dump_matrix(C, 1, n, "C");
    dump_matrix(O, m, n, "O");
    puts("");
  }

  // Warmup
  if (verbose > 1)
    puts("MLP\n");

  kernel(A, B, C, O, m, n, k);

  if (verbose > 1)
    dump_matrix(O, m, n, "O");

  // Run the reference benchmark
  if (verbose > 1)
    puts("BENCH");

  int skipped = 0;
  double* results = (double*)malloc(iter*sizeof(double));
  for (int i=0; i<iter; i++) {
    double start = clock();
    kernel(A, B, C, O, m, n, k);
    double stop = clock();
    double time = (stop - start)/CLOCKS_PER_SEC;
    // If counter is too low, we can't take this measurement
    if (time == 0.0) {
      skipped++;
      continue;
    }
    results[i-skipped] = gflops / time;
  }
  double mean = list_mean(results, iter-skipped);
  double stdev = list_stdev(results, iter-skipped, mean);

  printf("%6.3f +- %6.3f gflops\n", mean, stdev);

  free(A);
  free(B);
  free(C);
  free(O);

  return 0;
}
