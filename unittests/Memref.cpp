#include "../benchmarks/memref.h"
#include "gtest/gtest.h"

void init_matrix(struct vec_i2d *matrix) {
  for (int64_t mi = 0; mi < matrix->sizes[0]; ++mi) {
    for (int64_t ni = 0; ni < matrix->sizes[1]; ++ni) {
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

void init_4dtensor(struct vec_i4d *tensor) {
  for (int64_t p1idx = 0; p1idx < tensor->sizes[0]; p1idx++) {
    for (int64_t p2idx = 0; p2idx < tensor->sizes[1]; p2idx++) {
      for (int64_t p3idx = 0; p3idx < tensor->sizes[2]; p3idx++) {
        for (int64_t p4idx = 0; p4idx < tensor->sizes[3]; p4idx++) {
          vec_i4d_set(tensor, p1idx, p2idx, p3idx, p4idx,
                      p1idx + p2idx + p3idx + p4idx);
        }
      }
    }
  }
}

void init_3dtensor(struct vec_i3d *tensor) {
  for (int64_t p1idx = 0; p1idx < tensor->sizes[0]; p1idx++) {
    for (int64_t p2idx = 0; p2idx < tensor->sizes[1]; p2idx++) {
      for (int64_t p3idx = 0; p3idx < tensor->sizes[2]; p3idx++) {
        vec_i3d_set(tensor, p1idx, p2idx, p3idx, p1idx + p2idx + p3idx);
      }
    }
  }
}

TEST(memrefTest, memref1D) {
  // memref<23>
  const int m = 23;
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

TEST(memrefTest, memref2D) {
  // memref<23x12>
  const int m = 23;
  const int n = 12;
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

  for (int64_t mi = 0; mi < memref.sizes[0]; ++mi) {
    for (int64_t ni = 0; ni < memref.sizes[1]; ++ni) {
      EXPECT_EQ(vec_i2d_get(&memref, mi, ni), goldref[mi][ni]);
    }
  }
  vec_i2d_destroy(&memref);
}

TEST(memrefTest, memref3D) {
  // memref<89x99x2>
  const int m = 89;
  const int n = 99;
  const int l = 2;
  struct vec_i3d memref;
  memset(&memref, 0, sizeof(memref));
  vec_i3d_alloc(&memref, m, n, l);
  EXPECT_EQ(memref.sizes[0], 89);
  EXPECT_EQ(memref.sizes[1], 99);
  EXPECT_EQ(memref.sizes[2], 2);
  EXPECT_EQ(memref.strides[0], 198);
  EXPECT_EQ(memref.strides[1], 2);
  EXPECT_EQ(memref.strides[2], 1);
  init_3dtensor(&memref);

  int goldref[m][n][l];
  for (int64_t mi = 0; mi < m; mi++) {
    for (int64_t ni = 0; ni < n; ni++) {
      for (int64_t li = 0; li < l; li++) {
        goldref[mi][ni][li] = mi + ni + li;
      }
    }
  }

  for (int64_t mi = 0; mi < m; mi++) {
    for (int64_t ni = 0; ni < n; ni++) {
      for (int64_t li = 0; li < l; li++) {
        EXPECT_EQ(vec_i3d_get(&memref, mi, ni, li), goldref[mi][ni][li]);
      }
    }
  }
  vec_i3d_destroy(&memref);
}

TEST(memrefTest, memref4D) {
  // memref<23x12x18x99>
  const int m = 23;
  const int n = 12;
  const int l = 18;
  const int e = 99;
  struct vec_i4d memref;
  memset(&memref, 0, sizeof(memref));
  vec_i4d_alloc(&memref, m, n, l, e);
  EXPECT_EQ(memref.sizes[0], 23);
  EXPECT_EQ(memref.sizes[1], 12);
  EXPECT_EQ(memref.sizes[2], 18);
  EXPECT_EQ(memref.sizes[3], 99);
  EXPECT_EQ(memref.strides[3], 1);
  EXPECT_EQ(memref.strides[2], 99);
  EXPECT_EQ(memref.strides[1], 1782);
  EXPECT_EQ(memref.strides[0], 21384);
  init_4dtensor(&memref);

  int goldref[m][n][l][e];
  for (int64_t mi = 0; mi < m; mi++) {
    for (int64_t ni = 0; ni < n; ni++) {
      for (int64_t li = 0; li < l; li++) {
        for (int64_t ei = 0; ei < e; ei++) {
          goldref[mi][ni][li][ei] = mi + ni + li + ei;
        }
      }
    }
  }

  for (int64_t mi = 0; mi < memref.sizes[0]; mi++) {
    for (int64_t ni = 0; ni < memref.sizes[1]; ni++) {
      for (int64_t li = 0; li < memref.sizes[2]; li++) {
        for (int64_t ei = 0; ei < memref.sizes[3]; ei++) {
          EXPECT_EQ(vec_i4d_get(&memref, mi, ni, li, ei),
                    goldref[mi][ni][li][ei]);
        }
      }
    }
  }

  vec_i4d_destroy(&memref);
}
