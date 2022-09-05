#include "../memref.h"
#include <stdio.h>
#include <string.h>

/* Generated function under test */
extern void simple_copy(DECL_VEC2D_FUNC_IN_ARGS(input, float),
                        DECL_VEC2D_FUNC_OUT_ARGS(output, float));

/* Initialize matrix with value x+y at position (x, y) */
void init_matrix(struct vec_f2d *m) {
  for (int64_t y = 0; y < m->sizes[0]; y++)
    for (int64_t x = 0; x < m->sizes[1]; x++)
      vec_f2d_set(m, x, y, x + y);
}

void simple_copy_ref(const struct vec_f2d *input, struct vec_f2d *output) {
  for (int64_t y = 0; y < output->sizes[0]; y++) {
    for (int64_t x = 0; x < output->sizes[1]; x++) {
      float elem = vec_f2d_get(input, x, y);
      vec_f2d_set(output, x, y, elem);
    }
  }
}

int main(int argc, char **argv) {
  struct vec_f2d input, output, output_ref;
  int verbose = 1;
  int n = 6;
  int m = 9;

  if (vec_f2d_alloc(&input, n, m) || vec_f2d_alloc(&output, n, m) ||
      vec_f2d_alloc(&output_ref, n, m)) {
    fprintf(stderr, "Allocation failed");
    exit(1);
  }

  init_matrix(&input);

  if (verbose) {
    puts("input:");
    vec_f2d_dump(&input);
    puts("");
  }

  simple_copy(VEC2D_ARGS(&input), VEC2D_ARGS(&output));
  simple_copy_ref(&input, &output_ref);

  if (verbose) {
    puts("Result O:");
    vec_f2d_dump(&output);
    puts("");

    puts("Result O_ref:");
    vec_f2d_dump(&output_ref);
    puts("");
  }

  if (!vec_f2d_compare(&output, &output_ref)) {
    fputs("Result differs from reference result\n", stderr);
    exit(1);
  }

  vec_f2d_destroy(&input);
  vec_f2d_destroy(&output);
  vec_f2d_destroy(&output_ref);

  fputs("Result is correct\n", stderr);
  return 0;
}
