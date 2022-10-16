#ifndef MEMREF_H
#define MEMREF_H

// Taken from:
// https://github.com/andidr/teckyl/blob/master/tests/exec/lib/memref.h

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define DECL_VECND_STRUCT(ndims, name, eltype)                                 \
  struct name {                                                                \
    eltype *allocatedPtr;                                                      \
    eltype *alignedPtr;                                                        \
    int64_t offset;                                                            \
    int64_t sizes[ndims];                                                      \
    int64_t strides[ndims];                                                    \
  };

//===----------------------------------------------------------------------===//
// 1D memref
//===----------------------------------------------------------------------===//

/* 1D memref */
#define DECL_VEC1D_STRUCT(name, eltype) DECL_VECND_STRUCT(1, name, eltype)

/* Generates a comma-separated list of arguments from the fields of a
 * 1d memref */
#define VEC1D_ARGS(v)                                                          \
  (v)->allocatedPtr, (v)->alignedPtr, (v)->offset, (v)->sizes[0],              \
      (v)->strides[0]

/* Forwards all arguments declared with DECL_VEC1D_FUNC_ARGS to
 * another function */
#define VEC1D_FWD_ARGS(prefix)                                                 \
  prefix##_allocatedPtr, prefix##_alignedPtr, prefix##_offset, prefix##_size0, \
      prefix##_stride0

#define DECL_VEC1D_FUNC_ARGS(prefix, eltype, cst)                              \
  cst eltype *prefix##_allocatedPtr, cst eltype *prefix##_alignedPtr,          \
      int64_t prefix##_offset, int64_t prefix##_size0,                         \
      int64_t prefix##_stride0

#define DECL_VEC1D_FUNC_IN_ARGS(prefix, eltype)                                \
  DECL_VEC1D_FUNC_ARGS(prefix, eltype, const)

#define DECL_VEC1D_FUNC_OUT_ARGS(prefix, eltype)                               \
  DECL_VEC1D_FUNC_ARGS(prefix, eltype, )

/* Dumps the meta-information of a 1d memref to stdout. */
static inline void memref_1d_dump_metainfo(DECL_VEC1D_FUNC_ARGS(m, void,
                                                                const)) {
  printf("1d memref:\n"
         "  allocatedPtr: %p\n"
         "  alignedPtr: %p\n"
         "  offset: %" PRIu64 "\n"
         "  size0: %" PRIu64 "\n"
         "  stride0: %" PRIu64 "\n",
         m_allocatedPtr, m_alignedPtr, m_offset, m_size0, m_stride0);
}

#define DECL_VEC1D_FUNCTIONS(name, eltype, format)                             \
  /* Allocates and initializes a 1d memref. Returns 0 on success,              \
   * otherwise 1.                                                              \
   */                                                                          \
  static inline int name##_alloc(struct name *v, size_t n) {                   \
    eltype *f;                                                                 \
    if (!(f = (eltype *)calloc(n, sizeof(eltype))))                            \
      return 1;                                                                \
    v->allocatedPtr = f;                                                       \
    v->alignedPtr = f;                                                         \
    v->offset = 0;                                                             \
    v->sizes[0] = n;                                                           \
    v->strides[0] = 1;                                                         \
    return 0;                                                                  \
  }                                                                            \
  /* Destroys a 1d memref */                                                   \
  static inline void name##_destroy(struct name *v) { free(v->allocatedPtr); } \
  /* Returns the element at position (`x`) of a 1d memref `v` */               \
  static inline eltype name##_get(const struct name *v, int64_t x) {           \
    return *(v->alignedPtr + x);                                               \
  }                                                                            \
  /* Assigns `f` to the element at position (`x`) of a 1d                      \
   * memref `v`                                                                \
   */                                                                          \
  static inline void name##_set(struct name *v, int64_t x, eltype f) {         \
    v->alignedPtr[x] = f;                                                      \
  }

//===----------------------------------------------------------------------===//
// 2D memref
//===----------------------------------------------------------------------===//

/* 2D memref */
#define DECL_VEC2D_STRUCT(name, eltype) DECL_VECND_STRUCT(2, name, eltype)

/* Generates a comma-separated list of arguments from the fields of a
 * 2d memref */
#define VEC2D_ARGS(v)                                                          \
  (v)->allocatedPtr, (v)->alignedPtr, (v)->offset, (v)->sizes[0],              \
      (v)->sizes[1], (v)->strides[0], (v)->strides[1]

/* Forwards all arguments declared with DECL_VEC2D_FUNC_ARGS to
 * another function */
#define VEC2D_FWD_ARGS(prefix)                                                 \
  prefix##_allocatedPtr, prefix##_alignedPtr, prefix##_offset, prefix##_size0, \
      prefix##_size1, prefix##_stride0, prefix##_stride1

#define DECL_VEC2D_FUNC_ARGS(prefix, eltype, cst)                              \
  cst eltype *prefix##_allocatedPtr, cst eltype *prefix##_alignedPtr,          \
      int64_t prefix##_offset, int64_t prefix##_size0, int64_t prefix##_size1, \
      int64_t prefix##_stride0, int64_t prefix##_stride1

#define DECL_VEC2D_FUNC_IN_ARGS(prefix, eltype)                                \
  DECL_VEC2D_FUNC_ARGS(prefix, eltype, const)

#define DECL_VEC2D_FUNC_OUT_ARGS(prefix, eltype)                               \
  DECL_VEC2D_FUNC_ARGS(prefix, eltype, )

/* Dumps the meta-information of a 2d memref to stdout. */
static inline void memref_2d_dump_metainfo(DECL_VEC2D_FUNC_ARGS(m, void,
                                                                const)) {
  printf("2d memref:\n"
         "  allocatedPtr: %p\n"
         "  alignedPtr: %p\n"
         "  offset: %" PRIu64 "\n"
         "  size0: %" PRIu64 "\n"
         "  size1: %" PRIu64 "\n"
         "  stride0: %" PRIu64 "\n"
         "  stride1: %" PRIu64 "\n",
         m_allocatedPtr, m_alignedPtr, m_offset, m_size0, m_size1, m_stride0,
         m_stride1);
}

#define DECL_VEC2D_FUNCTIONS(name, eltype, format)                             \
  /* Allocates and initializes a 2d memref. Returns 0 on success,              \
   * otherwise 1.                                                              \
   */                                                                          \
  static inline int name##_alloc(struct name *v, size_t n, size_t m) {         \
    eltype *f;                                                                 \
    if (!(f = (eltype *)calloc(n * m, sizeof(eltype))))                        \
      return 1;                                                                \
    v->allocatedPtr = f;                                                       \
    v->alignedPtr = f;                                                         \
    v->offset = 0;                                                             \
    v->sizes[0] = n;                                                           \
    v->sizes[1] = m;                                                           \
    v->strides[0] = m;                                                         \
    v->strides[1] = 1;                                                         \
    return 0;                                                                  \
  }                                                                            \
  /* Destroys a 2d memref */                                                   \
  static inline void name##_destroy(struct name *v) { free(v->allocatedPtr); } \
  /* Returns the element at position (`p1`, `p2`) of a 2d memref `v` */        \
  static inline eltype name##_get(const struct name *v, int64_t p1,            \
                                  int64_t p2) {                                \
    return *(v->alignedPtr + p1 * v->sizes[1] + p2);                           \
  }                                                                            \
  /* Assigns `f` to the element at position (`p1`, `p2`) of a 2d               \
   * memref `v`                                                                \
   */                                                                          \
  static inline void name##_set(struct name *v, int64_t p1, int64_t p2,        \
                                eltype f) {                                    \
    *(v->alignedPtr + p1 * v->sizes[1] + p2) = f;                              \
  }                                                                            \
  /* Dumps a 2d `v` to stdout. */                                              \
  static inline void name##_dump(const struct name *v) {                       \
    for (int64_t y = 0; y < v->sizes[0]; y++) {                                \
      for (int64_t x = 0; x < v->sizes[1]; x++) {                              \
        printf(format "%s", *(v->alignedPtr + y * v->sizes[1] + x),            \
               x == v->sizes[1] - 1 ? "" : " ");                               \
      }                                                                        \
      puts("");                                                                \
    }                                                                          \
  }

//===----------------------------------------------------------------------===//
// 3D memref
//===----------------------------------------------------------------------===//

/* 3D memref */
#define DECL_VEC3D_STRUCT(name, eltype) DECL_VECND_STRUCT(3, name, eltype)

/* Generates a comma-separated list of arguments from the fields of a
 * 3d memref */
#define VEC3D_ARGS(v)                                                          \
  (v)->allocatedPtr, (v)->alignedPtr, (v)->offset, (v)->sizes[0],              \
      (v)->sizes[1], (v)->sizes[2], (v)->strides[0], (v)->strides[1],          \
      (v)->strides[2]

/* Forwards all arguments declared with DECL_VEC3D_FUNC_ARGS to
 * another function */
#define VEC3D_FWD_ARGS(prefix)                                                 \
  prefix##_allocatedPtr, prefix##_alignedPtr, prefix##_offset, prefix##_size0, \
      prefix##_size1, prefix##_size2, prefix##_stride0, prefix##_stride1,      \
      prefix##_stride2

#define DECL_VEC3D_FUNC_ARGS(prefix, eltype, cst)                              \
  cst eltype *prefix##_allocatedPtr, cst eltype *prefix##_alignedPtr,          \
      int64_t prefix##_offset, int64_t prefix##_size0, int64_t prefix##_size1, \
      int64_t prefix##_size2, int64_t prefix##_stride0,                        \
      int64_t prefix##_stride1, int64_t prefix##_stride2

#define DECL_VEC3D_FUNC_IN_ARGS(prefix, eltype)                                \
  DECL_VEC3D_FUNC_ARGS(prefix, eltype, const)

#define DECL_VEC3D_FUNC_OUT_ARGS(prefix, eltype)                               \
  DECL_VEC3D_FUNC_ARGS(prefix, eltype, )

/* Dumps the meta-information of a 3d memref to stdout. */
static inline void memref_3d_dump_metainfo(DECL_VEC3D_FUNC_ARGS(m, void,
                                                                const)) {
  printf("3d memref:\n"
         "  allocatedPtr: %p\n"
         "  alignedPtr: %p\n"
         "  offset: %" PRIu64 "\n"
         "  size0: %" PRIu64 "\n"
         "  size1: %" PRIu64 "\n"
         "  size2: %" PRIu64 "\n"
         "  stride0: %" PRIu64 "\n"
         "  stride1: %" PRIu64 "\n"
         "  stride2: %" PRIu64 "\n",
         m_allocatedPtr, m_alignedPtr, m_offset, m_size0, m_size1, m_size2,
         m_stride0, m_stride1, m_stride2);
}

#define DECL_VEC3D_FUNCTIONS(name, eltype, format)                             \
  /* Allocates and initializes a 3d memref. Returns 0 on success,              \
   * otherwise 1.                                                              \
   */                                                                          \
  static inline int name##_alloc(struct name *v, size_t n, size_t m,           \
                                 size_t l) {                                   \
    eltype *f;                                                                 \
    if (!(f = (eltype *)calloc(n * m * l, sizeof(eltype))))                    \
      return 1;                                                                \
    v->allocatedPtr = f;                                                       \
    v->alignedPtr = f;                                                         \
    v->offset = 0;                                                             \
    v->sizes[0] = n;                                                           \
    v->sizes[1] = m;                                                           \
    v->sizes[2] = l;                                                           \
    v->strides[0] = m * l;                                                     \
    v->strides[1] = l;                                                         \
    v->strides[2] = 1;                                                         \
    return 0;                                                                  \
  }                                                                            \
  /* Destroys a 3d memref */                                                   \
  static inline void name##_destroy(struct name *v) { free(v->allocatedPtr); } \
  /* Returns the element at position (`p1`, `p2`, `p3`) of a 3d memref         \
   * `v`                                                                       \
   */                                                                          \
  static inline eltype name##_get(const struct name *v, int64_t p1,            \
                                  int64_t p2, int64_t p3) {                    \
    return *(v->alignedPtr + (p1 * v->sizes[1] * v->sizes[2]) +                \
             (p2 * v->sizes[2]) + p3);                                         \
  }                                                                            \
  /* Assigns `f` to the element at position (`p1`, `p2`, `p3`) of a 3d         \
   * memref `v`                                                                \
   */                                                                          \
  static inline void name##_set(struct name *v, int64_t p1, int64_t p2,        \
                                int64_t p3, eltype f) {                        \
    *(v->alignedPtr + (p1 * v->sizes[1] * v->sizes[2]) + (p2 * v->sizes[2]) +  \
      p3) = f;                                                                 \
  }

//===----------------------------------------------------------------------===//
// 4D memref
//===----------------------------------------------------------------------===//

/* 4D memref */
#define DECL_VEC4D_STRUCT(name, eltype) DECL_VECND_STRUCT(4, name, eltype)

/* Generates a comma-separated list of arguments from the fields of a
 * 4d memref */
#define VEC4D_ARGS(v)                                                          \
  (v)->allocatedPtr, (v)->alignedPtr, (v)->offset, (v)->sizes[0],              \
      (v)->sizes[1], (v)->sizes[2], (v)->sizes[3], (v)->strides[0],            \
      (v)->strides[1], (v)->strides[2], (v)->strides[3]

/* Forwards all arguments declared with DECL_VEC4D_FUNC_ARGS to
 * another function */
#define VEC4D_FWD_ARGS(prefix)                                                 \
  prefix##_allocatedPtr, prefix##_alignedPtr, prefix##_offset, prefix##_size0, \
      prefix##_size1, prefix##_size2, prefix##_size3, prefix##_stride0,        \
      prefix##_stride1, prefix##_stride2, prefix##_stride3

#define DECL_VEC4D_FUNC_ARGS(prefix, eltype, cst)                              \
  cst eltype *prefix##_allocatedPtr, cst eltype *prefix##_alignedPtr,          \
      int64_t prefix##_offset, int64_t prefix##_size0, int64_t prefix##_size1, \
      int64_t prefix##_size2, int64_t prefix##_size3,                          \
      int64_t prefix##_stride0, int64_t prefix##_stride1,                      \
      int64_t prefix##_stride2, int64_t prefix##_stride3

#define DECL_VEC4D_FUNC_IN_ARGS(prefix, eltype)                                \
  DECL_VEC4D_FUNC_ARGS(prefix, eltype, const)

#define DECL_VEC4D_FUNC_OUT_ARGS(prefix, eltype)                               \
  DECL_VEC4D_FUNC_ARGS(prefix, eltype, )

/* Dumps the meta-information of a 4d memref to stdout. */
static inline void memref_4d_dump_metainfo(DECL_VEC4D_FUNC_ARGS(m, void,
                                                                const)) {
  printf("4d memref:\n"
         "  allocatedPtr: %p\n"
         "  alignedPtr: %p\n"
         "  offset: %" PRIu64 "\n"
         "  size0: %" PRIu64 "\n"
         "  size1: %" PRIu64 "\n"
         "  size2: %" PRIu64 "\n"
         "  size3: %" PRIu64 "\n"
         "  stride0: %" PRIu64 "\n"
         "  stride1: %" PRIu64 "\n"
         "  stride2: %" PRIu64 "\n"
         "  stride3: %" PRIu64 "\n",
         m_allocatedPtr, m_alignedPtr, m_offset, m_size0, m_size1, m_size2,
         m_size3, m_stride0, m_stride1, m_stride2, m_stride3);
}

#define DECL_VEC4D_FUNCTIONS(name, eltype, format)                             \
  /* Allocates and initializes a 4d memref. Returns 0 on success,              \
   * otherwise 1.                                                              \
   */                                                                          \
  static inline int name##_alloc(struct name *v, size_t n, size_t m, size_t l, \
                                 size_t e) {                                   \
    eltype *f;                                                                 \
    if (!(f = (eltype *)calloc(n * m * l * e, sizeof(eltype))))                \
      return 1;                                                                \
    v->allocatedPtr = f;                                                       \
    v->alignedPtr = f;                                                         \
    v->offset = 0;                                                             \
    v->sizes[0] = n;                                                           \
    v->sizes[1] = m;                                                           \
    v->sizes[2] = l;                                                           \
    v->sizes[3] = e;                                                           \
    v->strides[0] = m * l * e;                                                 \
    v->strides[1] = l * e;                                                     \
    v->strides[2] = e;                                                         \
    v->strides[3] = 1;                                                         \
    return 0;                                                                  \
  }                                                                            \
  /* Destroys a 4d memref */                                                   \
  static inline void name##_destroy(struct name *v) { free(v->allocatedPtr); } \
  /* Returns the element at position (`p1`, `p2`, `p3`, `p4`) of a 4d memref   \
   * `v`                                                                       \
   */                                                                          \
  static inline eltype name##_get(const struct name *v, int64_t p1,            \
                                  int64_t p2, int64_t p3, int64_t p4) {        \
    return *(v->alignedPtr + (p1 * v->sizes[1] * v->sizes[2] * v->sizes[3]) +  \
             (p2 * v->sizes[2] * v->sizes[3]) + (p3 * v->sizes[3]) + p4);      \
  }                                                                            \
  /* Assigns `f` to the element at position (`p1`, `p2`, `p3`, `p4`) of a 4d   \
   * memref `v`                                                                \
   */                                                                          \
  static inline void name##_set(struct name *v, int64_t p1, int64_t p2,        \
                                int64_t p3, int64_t p4, eltype f) {            \
    *(v->alignedPtr + (p1 * v->sizes[1] * v->sizes[2] * v->sizes[3]) +         \
      (p2 * v->sizes[2] * v->sizes[3]) + (p3 * v->sizes[3]) + p4) = f;         \
  }

DECL_VEC4D_STRUCT(vec_f4d, float)
DECL_VEC4D_FUNCTIONS(vec_f4d, float, "%f")
DECL_VEC4D_STRUCT(vec_i4d, int)
DECL_VEC4D_FUNCTIONS(vec_i4d, int, "%i")

DECL_VEC3D_STRUCT(vec_f3d, float)
DECL_VEC3D_FUNCTIONS(vec_f3d, float, "%f")
DECL_VEC3D_STRUCT(vec_i3d, int)
DECL_VEC3D_FUNCTIONS(vec_i3d, int, "%i")

DECL_VEC2D_STRUCT(vec_f2d, float)
DECL_VEC2D_FUNCTIONS(vec_f2d, float, "%f")
DECL_VEC2D_STRUCT(vec_i2d, int)
DECL_VEC2D_FUNCTIONS(vec_i2d, int, "%i")

DECL_VEC1D_STRUCT(vec_f1d, float)
DECL_VEC1D_FUNCTIONS(vec_f1d, float, "%f")
DECL_VEC1D_STRUCT(vec_i1d, int)
DECL_VEC1D_FUNCTIONS(vec_i1d, int, "%i")

#endif
