#include "../benchmarks/memref.h"
#include "gtest/gtest.h"

void init_matrix(struct vec_i2d *matrix) {
  const int64_t m = matrix->sizes[1], n = matrix->sizes[0];
  for (int64_t mi = 0; mi < m; ++mi) {
    for (int64_t ni = 0; ni < n; ++ni) {
      vec_i2d_set(matrix, mi, ni, mi + ni);
    }
  }
}

void init_matrix(struct vec_i1d *vector) {
  const int64_t m = vector->sizes[0];
  for (int64_t mi = 0; mi < m; ++mi) {
    vec_i1d_set(vector, mi, mi + mi);
  }
}

TEST(memrefTest, memref2D) {
  // memref<23x12>
  int m = 23;
  int n = 12;
  struct vec_i2d memref;
  memset(&memref, 0, sizeof(memref));
  vec_i2d_alloc(&memref, m, n);
  EXPECT_EQ(memref.sizes[0], 23);
  EXPECT_EQ(memref.sizes[1], 12);
  EXPECT_EQ(memref.strides[0], 12);
  EXPECT_EQ(memref.strides[1], 1);
  init_matrix(&memref);

  int goldref[m][n];
  for (int64_t row = 0; row < m; row++) {
    for (int64_t col = 0; col < n; col++) {
      goldref[row][col] = row + col;
    }
  }

  for (int64_t mi = 0; mi < memref.sizes[1]; ++mi) {
    for (int64_t ni = 0; ni < memref.sizes[0]; ++ni) {
      EXPECT_EQ(vec_i2d_get(&memref, mi, ni), goldref[ni][mi]);
    }
  }
  vec_i2d_destroy(&memref);
}

TEST(memrefTest, memref1D) {
  // memref<23>
  int m = 23;
  struct vec_i1d memref;
  memset(&memref, 0, sizeof(memref));
  vec_i1d_alloc(&memref, m);
  EXPECT_EQ(memref.sizes[0], 23);
  EXPECT_EQ(memref.strides[0], 1);
  init_matrix(&memref);

  int goldref[m];
  for (int64_t row = 0; row < m; row++) {
    goldref[row] = row + row;
  }

  for (int64_t row = 0; row < m; row++) {
    EXPECT_EQ(vec_i1d_get(&memref, row), goldref[row]);
  }
  vec_i1d_destroy(&memref);
}
