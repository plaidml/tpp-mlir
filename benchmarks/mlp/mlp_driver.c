#include "../memref.h"
#include <stdio.h>
#include <string.h>

#include <libxsmm.h>

#define MAX(a,b) ((a) > (b) ? (a) : (b))

/* Generated mlp under test */
extern void mlp(DECL_VEC2D_FUNC_IN_ARGS(a, float),
                DECL_VEC2D_FUNC_IN_ARGS(b, float),
                DECL_VEC2D_FUNC_OUT_ARGS(c, float),
                DECL_VEC2D_FUNC_OUT_ARGS(out, float));

/* Reference implementation of a mlp */
// a =   128 x 256
// b =   256 x 512
// c =   1 x 512
// out = 128x512

void mlp_refimpl(const struct vec_f2d *a, const struct vec_f2d *b,
                 struct vec_f2d *c, struct vec_f2d *out) {
  const int64_t m = out->sizes[1], n = out->sizes[0], k = a->sizes[1];
  // BROADCAST
  for (int64_t ni = 0; ni < n; ++ni) {
    for (int64_t mi = 0; mi < m; ++mi) {
      float tmp = vec_f2d_get(c, mi, 0);
      //printf("%f\n", tmp);
      vec_f2d_set(out, mi, ni, tmp);
    }
  }
  //puts("----");
  //printf("%" PRId64 "\n", m);
  //vec_f2d_dump(out);
  //puts("----");

  // MUL
  for (int64_t ni = 0; ni < n; ++ni) {
    for (int64_t ki = 0; ki < k; ++ki) {
      for (int64_t mi = 0; mi < m; ++mi) {
        vec_f2d_set(out, mi, ni,
                    vec_f2d_get(a, ki, ni) * vec_f2d_get(b, mi, ki) +
                        vec_f2d_get(out, mi, ni)); // beta=1
      }
    }
  }

  // RELU 
  for (int64_t ni = 0; ni < n; ++ni) {
    for (int64_t mi = 0; mi < m; mi++) {
      vec_f2d_set(out, mi, ni, MAX(0, vec_f2d_get(out, mi, ni)));
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

/* Clear matrix (aka fill with zeros) */
void clear_matrix(struct vec_f2d *matrix) {
  const int64_t m = matrix->sizes[1], n = matrix->sizes[0];
  for (int64_t mi = 0; mi < m; ++mi) {
    for (int64_t ni = 0; ni < n; ++ni) {
      vec_f2d_set(matrix, mi, ni, 0);
    }
  }
}

int main(int argc, char *argv[]) {

  struct vec_f2d a, b, c, out, out_ref;
  memset(&a, 0, sizeof(a));
  memset(&b, 0, sizeof(b));
  memset(&c, 0, sizeof(c));
  memset(&out, 0, sizeof(out));
  memset(&out_ref, 0, sizeof(out_ref));

  //int m = 128;
  //int n = 512;
  //int k = 256;
  int m = 4;
  int n = 16;
  int k = 8;
  int verbose = 1;

  if (EXIT_SUCCESS == vec_f2d_alloc(&a, m, k) &&
      EXIT_SUCCESS == vec_f2d_alloc(&b, k, n) &&
      EXIT_SUCCESS == vec_f2d_alloc(&c, 1, n) &&
      EXIT_SUCCESS == vec_f2d_alloc(&out, m, n) &&
      EXIT_SUCCESS == vec_f2d_alloc(&out_ref, m, n)) {
    init_matrix(&a);
    init_matrix(&b);
    init_matrix(&c);

    if (0 != verbose) {
      puts("C:");
      vec_f2d_dump(&c);
      puts("");
    }

    // validate results
    clear_matrix(&out);
    clear_matrix(&out_ref);
    mlp(VEC2D_ARGS(&a), VEC2D_ARGS(&b), VEC2D_ARGS(&c), VEC2D_ARGS(&out));
    mlp_refimpl(&a, &b, &c, &out_ref);

    if (0 != verbose) {
      puts("Result:");
      vec_f2d_dump(&out);
      puts("");

      puts("Reference:");
      vec_f2d_dump(&out_ref);
      puts("");
    }

    if (0 == vec_f2d_compare(&out, &out_ref)) {
      libxsmm_matdiff_info diff;
      double error = 0.0;
      libxsmm_matdiff(&diff, LIBXSMM_DATATYPE_F32, n, m, out.alignedPtr,
                      out_ref.alignedPtr, &n, &n);
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
        return EXIT_FAILURE;
      }
    }
  }

  vec_f2d_destroy(&a);
  vec_f2d_destroy(&b);
  vec_f2d_destroy(&c);
  vec_f2d_destroy(&out);
  vec_f2d_destroy(&out_ref);

  return EXIT_SUCCESS;
}
