#include "../memref.h"
#include <stdio.h>
#include <string.h>

#include <libxsmm.h>

/* Generated matrix multiplication function under test */
extern void matmul(DECL_VEC2D_FUNC_IN_ARGS(a, float),
                   DECL_VEC2D_FUNC_IN_ARGS(b, float),
                   DECL_VEC2D_FUNC_OUT_ARGS(o, float));

/* Reference implementation of a matrix multiplication */
void matmul_refimpl(const struct vec_f2d *a, const struct vec_f2d *b,
                    struct vec_f2d *o) {
  float accu;

  for (int64_t y = 0; y < o->sizes[0]; y++) {
    for (int64_t x = 0; x < o->sizes[1]; x++) {
      accu = 0;

      for (int64_t k = 0; k < a->sizes[1]; k++)
        accu += vec_f2d_get(a, k, y) * vec_f2d_get(b, x, k);

      vec_f2d_set(o, x, y, accu);
    }
  }
}

/* Initialize matrix with value x+y at position (x, y) */
void init_matrix(struct vec_f2d *m) {
  for (int64_t y = 0; y < m->sizes[0]; y++)
    for (int64_t x = 0; x < m->sizes[1]; x++)
      vec_f2d_set(m, x, y, x + y);
}

/* Clear matrix (aka fill with zeros */
void clear_matrix(struct vec_f2d *m) {
  for (int64_t y = 0; y < m->sizes[0]; y++)
    for (int64_t x = 0; x < m->sizes[1]; x++)
      vec_f2d_set(m, x, y, 0);
}

int main(int argc, char **argv) {
  struct vec_f2d a, b, o, o_ref;
  int verbose = 1;
  int n = 64;
  int k = 64;
  int m = 64;

  if (vec_f2d_alloc(&a, m, k) || vec_f2d_alloc(&b, k, n) ||
      vec_f2d_alloc(&o, m, n) || vec_f2d_alloc(&o_ref, m, n)) {
    fprintf(stderr, "Allocation failed");
    return 1;
  }

  init_matrix(&a);
  init_matrix(&b);

  if (verbose) {
    puts("A:");
    vec_f2d_dump(&a);
    puts("");

    puts("B:");
    vec_f2d_dump(&b);
    puts("");

    puts("O:");
    vec_f2d_dump(&o);
    puts("");
  }

  // preheating runs
  for (int i = 0; i < 5; i++)
    matmul(VEC2D_ARGS(&a), VEC2D_ARGS(&b), VEC2D_ARGS(&o));

  // actual runs
  libxsmm_timer_tickint start = libxsmm_timer_tick();
  for (int i = 0; i < 20; i++)
    matmul(VEC2D_ARGS(&a), VEC2D_ARGS(&b), VEC2D_ARGS(&o));
  libxsmm_timer_tickint stop = libxsmm_timer_tick();
  double duration = libxsmm_timer_duration(start, stop);
  printf("Duration LIBXSMM: %f\n", duration);

  clear_matrix(&o);
  clear_matrix(&o_ref);
  matmul(VEC2D_ARGS(&a), VEC2D_ARGS(&b), VEC2D_ARGS(&o));
  matmul_refimpl(&a, &b, &o_ref);

  if (verbose) {
    puts("Result O:");
    vec_f2d_dump(&o);
    puts("");

    puts("Reference O:");
    vec_f2d_dump(&o_ref);
    puts("");
  }

  libxsmm_matdiff_info l_diff;
  double error = 0.0;
  libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F32, m, n, o_ref.allocatedPtr,
                  o.allocatedPtr, &n, &n);
  error = libxsmm_matdiff_epsilon(&l_diff);

  printf("\nPrinting Norms:\n");
  printf("L1 reference  : %.25g\n", l_diff.l1_ref);
  printf("L1 test       : %.25g\n", l_diff.l1_tst);
  printf("L2 abs.error  : %.24f\n", l_diff.l2_abs);
  printf("L2 rel.error  : %.24f\n", l_diff.l2_rel);
  printf("Linf abs.error: %.24f\n", l_diff.linf_abs);
  printf("Linf rel.error: %.24f\n", l_diff.linf_rel);
  printf("Check-norm    : %.24f\n", error);
  printf("\n");

  vec_f2d_destroy(&a);
  vec_f2d_destroy(&b);
  vec_f2d_destroy(&o);
  vec_f2d_destroy(&o_ref);

  if (error > 0.000005)
    fputs("Error too high\n", stderr);
  else
    fputs("Result is correct\n", stderr);
  return 0;
}
