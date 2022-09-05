#include "../memref.h"
#include <stdio.h>
#include <string.h>

#include <libxsmm.h>

#if !defined(ARG_MNK)
#define ARG_MNK "32x32x32"
#elif !defined(MLIR)
#define MLIR
#endif

/* Generated matrix multiplication function under test */
extern void matmul(DECL_VEC2D_FUNC_IN_ARGS(a, float),
                   DECL_VEC2D_FUNC_IN_ARGS(b, float),
                   DECL_VEC2D_FUNC_OUT_ARGS(c, float));

/* Reference implementation of a matrix multiplication */
void matmul_refimpl(const struct vec_f2d *a, const struct vec_f2d *b,
                    struct vec_f2d *c) {
  const int64_t m = c->sizes[1], n = c->sizes[0], k = a->sizes[1];
  for (int64_t ni = 0; ni < n; ++ni) {
    for (int64_t ki = 0; ki < k; ++ki) {
      for (int64_t mi = 0; mi < m; ++mi) {
        vec_f2d_set(c, mi, ni,
                    vec_f2d_get(a, ki, ni) * vec_f2d_get(b, mi, ki) +
                        vec_f2d_get(c, mi, ni)); // beta=1
      }
    }
  }
}

/* Initialize matrix with value x+y at position (x, y) */
void init_matrix(struct vec_f2d *matrix) {
  const int64_t m = matrix->sizes[1], n = matrix->sizes[0];
  for (int64_t mi = 0; mi < m; ++mi) {
    for (int64_t ni = 0; ni < n; ++ni) {
      vec_f2d_set(matrix, mi, ni, mi + ni);
    }
  }
}

/* Clear matrix (aka fill with zeros */
void clear_matrix(struct vec_f2d *matrix) {
  const int64_t m = matrix->sizes[1], n = matrix->sizes[0];
  for (int64_t mi = 0; mi < m; ++mi) {
    for (int64_t ni = 0; ni < n; ++ni) {
      vec_f2d_set(matrix, mi, ni, 0);
    }
  }
}

int main(int argc, char *argv[]) {
  struct vec_f2d a, b, c, d;
  const double max_duration = 5.0, max_epsilon = 5e-6;
  const int nwarmup = 10;
  int result = EXIT_SUCCESS;
  int nrepeat = (1 < argc ? atoi(argv[1]) : 0);
  const char *mnk = ((2 < argc && 0 < atoi(argv[2])) ? argv[2] : (ARG_MNK));
  const int verbose = (3 < argc ? atoi(argv[3]) : 0);
  int m = atoi(mnk), n = m, k = m;
  const char *x = strchr(mnk, 'x');

  // parse MxNxK
  if (NULL != x) {
    n = atoi(++x);
    x = strchr(x, 'x');
    if (NULL != x) {
      k = atoi(++x);
    }
  }

  // initialize (vec_f2d_destroy)
  memset(&a, 0, sizeof(a));
  memset(&b, 0, sizeof(b));
  memset(&c, 0, sizeof(c));
  memset(&d, 0, sizeof(d));

  if (EXIT_SUCCESS == vec_f2d_alloc(&a, m, k) &&
      EXIT_SUCCESS == vec_f2d_alloc(&b, k, n) &&
      EXIT_SUCCESS == vec_f2d_alloc(&c, m, n) &&
      EXIT_SUCCESS == vec_f2d_alloc(&d, m, n)) {
    init_matrix(&a);
    init_matrix(&b);

    if (0 != verbose) {
      puts("A:");
      vec_f2d_dump(&a);
      puts("");

      puts("B:");
      vec_f2d_dump(&b);
      puts("");

      if (2 <= verbose || 0 > verbose) {
        puts("C:");
        vec_f2d_dump(&c);
        puts("");
      }
    }

    // warmup and calibration for max_duration
    libxsmm_timer_tickint start = libxsmm_timer_tick();
    for (int i = 0; i < nwarmup; i++) {
      matmul(VEC2D_ARGS(&a), VEC2D_ARGS(&b), VEC2D_ARGS(&c));
    }
    double duration = libxsmm_timer_duration(start, libxsmm_timer_tick());

    if (0 >= nrepeat) {
      nrepeat = (0 < duration ? (int)LIBXSMM_ROUND(max_duration / duration)
                              : nwarmup);
    }
    if (nwarmup > nrepeat) {
      nrepeat = nwarmup;
    }

    // actual runs
    start = libxsmm_timer_tick();
    for (int i = 0; i < nrepeat; i++) {
      matmul(VEC2D_ARGS(&a), VEC2D_ARGS(&b), VEC2D_ARGS(&c));
    }
    duration = libxsmm_timer_duration(start, libxsmm_timer_tick());

    // validate result against reference
    clear_matrix(&c);
    clear_matrix(&d);
    matmul(VEC2D_ARGS(&a), VEC2D_ARGS(&b), VEC2D_ARGS(&c));
    matmul_refimpl(&a, &b, &d);

    if (0 != verbose) {
      puts("Result:");
      vec_f2d_dump(&c);
      puts("");

      puts("Reference:");
      vec_f2d_dump(&d);
      puts("");
    }

    if (0 == vec_f2d_compare(&c, &d) || 3 <= verbose || 0 > verbose) {
      libxsmm_matdiff_info diff;
      double error = 0.0;
      libxsmm_matdiff(&diff, LIBXSMM_DATATYPE_F32, n, m, d.alignedPtr,
                      c.alignedPtr, &n, &n);
      error = libxsmm_matdiff_epsilon(&diff);

      printf("Printing Norms:\n");
      printf("L1 reference  : %.25g\n", diff.l1_ref);
      printf("L1 test       : %.25g\n", diff.l1_tst);
      printf("L2 abs.error  : %.24f\n", diff.l2_abs);
      printf("L2 rel.error  : %.24f\n", diff.l2_rel);
      printf("Linf abs.error: %.24f\n", diff.linf_abs);
      printf("Linf rel.error: %.24f\n", diff.linf_rel);
      printf("Check-norm    : %.24f\n", error);
      printf("\n");

      if (0 < error) {
        fputs("Result differs from reference result\n", stderr);
        if (max_epsilon < error) {
          fputs("Error exceeds margin\n", stderr);
          result = EXIT_FAILURE;
        }
      }
    }

    if (EXIT_SUCCESS == result) {
      printf(
#if defined(MLIR)
          "LIBXSMM MLIR"
#else
          "LIBXSMM"
#endif
          ": %f GFLOPS/s\n",
          1e-9 * (2.0 * m * n * k * nrepeat) / duration);
      fputs("Result is correct\n", stderr);
    }
  } else {
    fprintf(stderr, "Allocation failed");
    result = EXIT_FAILURE;
  }

  vec_f2d_destroy(&a);
  vec_f2d_destroy(&b);
  vec_f2d_destroy(&c);
  vec_f2d_destroy(&d);

  return result;
}
