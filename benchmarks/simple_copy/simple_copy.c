#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "../utils.h"

void simple_copy_ref(const float *input, float *output, int n, int m) {
  for (int64_t y = 0; y < n; y++) {
    for (int64_t x = 0; x < m; x++) {
      output[y*n+x] = input[y*n+x];
    }
  }
}

void simple_copy_xsmm(const float *input, float *output, int n, int m) {
  assert(0 && "Not implemented yet");
}

int main(int argc, char **argv) {
  float* input;
  float* output;
  int verbose = 0;
  void(*kernel)(const float*, float*,int,int) = simple_copy_ref;
  // These need to be the same as the MLIR file
  int n = 1024;
  int m = 1024;
  int iter = 1000;

  // Cmd-line args
  int opt;
  while ((opt = getopt(argc, argv, "i:xv")) != -1) {
    switch (opt) {
      case 'i':
        iter = atoi(optarg);
        break;
      case 'x':
        kernel = simple_copy_xsmm;
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
  double gflops = (double)(n*m)/1e9;
  if (verbose) {
    printf("Kernel version: %s\n", (kernel == simple_copy_ref) ? "ref" : "xsmm");
    printf("[ %d, %d ] x %d\n", n, m, iter);
    printf("%lf Gfps\n", gflops);
  }

  // Initialize input
  input = (float*)malloc(n*m*sizeof(float));
  output = (float*)malloc(n*m*sizeof(float));
  if (!input || !output) {
    fprintf(stderr, "Allocation failed");
    exit(1);
  }
  init_matrix(input, n, m);
  if (verbose > 1) {
    dump_matrix(input, n, m, "input");
    puts("");
  }

  // Warmup
  if (verbose > 1)
    puts("COPY");

  kernel(input, output, n, m);

  if (verbose > 1)
    dump_matrix(output, n, m, "output");

  // Run the reference benchmark
  if (verbose > 1)
    puts("BENCH");

  int skipped = 0;
  double* results = (double*)malloc(iter*sizeof(double));
  for (int i=0; i<iter; i++) {
    double start = clock();
    kernel(input, output, m, n);
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

  free(input);
  free(output);

  return 0;
}
