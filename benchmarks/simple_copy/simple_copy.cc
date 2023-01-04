#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "../utils.h"
#include "../Perf.h"

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
  while ((opt = getopt(argc, argv, "n:xv")) != -1) {
    switch (opt) {
      case 'n':
        iter = atoi(optarg);
        break;
      case 'x':
        kernel = simple_copy_xsmm;
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
  double gflops = (double)(n*m)/1e9;
  if (verbose) {
    fprintf(stderr, "Kernel version: %s\n", (kernel == simple_copy_ref) ? "ref" : "xsmm");
    fprintf(stderr, "[ %d, %d ] x %d\n", n, m, iter);
    fprintf(stderr, "%lf Gfps\n", gflops);
  }

  // Initialize input
  input = (float*)malloc(n*m*sizeof(float));
  output = (float*)malloc(n*m*sizeof(float));
  if (!input || !output) {
    fprintf(stderr, "Allocation failed");
    exit(1);
  }
  init_matrix(input, n, m);
  if (verbose > 2) {
    dump_matrix(input, n, m, "input");
    puts("");
  }

  // Warmup
  if (verbose > 2)
    puts("COPY");

  kernel(input, output, n, m);

  if (verbose > 2)
    dump_matrix(output, n, m, "output");

  // Run the reference benchmark
  if (verbose > 2)
    puts("BENCH");

  PerfResults perf;
  for (int i=0; i<iter; i++) {
    perf.startTimer();
    kernel(input, output, m, n);
    perf.stopTimer();
  }
  double mean = perf.getMean();
  double stdev = perf.getStdev();

  if (verbose)
    fprintf(stderr, "%3.9f +- %3.9f ms\n", mean, stdev);
  printf("%9.3f +- %9.3f gflops\n", gflops/mean, (gflops*stdev)/(mean*mean));

  free(input);
  free(output);

  return 0;
}
