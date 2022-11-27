#ifndef _UTILS_H
#define _UTILS_H

#include <stdio.h>

#define MAX(a,b) ((a) > (b) ? (a) : (b))

static void dump_matrix(const float *input, int m, int n, const char* name) {
  printf("%s:\n[\n", name);
  for (int mi = 0; mi < m; mi++) {
    printf("[ ");
    for (int ni = 0; ni < n; ni++) {
      printf("%6.2f ", input[mi*n+ni]);
    }
    printf(" ]\n");
  }
  printf("]\n");
}

static void constant_matrix(float *input, int m, int n, float val) {
  for (int mi = 0; mi < m; mi++) {
    for (int ni = 0; ni < n; ni++) {
      input[mi*n+ni] = val;
    }
  }
}

static void init_matrix(float *input, int m, int n) {
  constant_matrix(input, m, n, 1.0);
}

static void clear_matrix(float *input, int m, int n) {
  constant_matrix(input, m, n, 0.0);
}

static void matmul_ref(const float *A, const float* B, float* O, int m, int n, int k) {
  // MATMUL O += A x B
  for (int mi = 0; mi < m; ++mi) {
    for (int ni = 0; ni < n; ++ni) {
      for (int ki = 0; ki < k; ++ki) {
        O[mi*n+ni] += A[mi*k+ki] * B[ki*n+ni];
      }
    }
  }
}

static double list_mean(const double* list, int len) {
  double sum = 0.0;
  for (int i=0; i<len; i++) {
    sum += list[i];
  }
  return sum/len;
}

static double list_stdev(const double* list, int len, double mean) {
  double sum = 0.0;
  for (int i=0; i<len; i++) {
    double delta = list[i]-mean;
    sum += delta*delta;
  }
  return sum/len;
}

#endif // _UTILS_H
