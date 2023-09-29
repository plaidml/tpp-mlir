// RUN: ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0:${ASAN_OPTIONS} \
// RUN: tpp-run %s -gpu=vulkan -print -seed 123 \
// RUN:  -entry-point-result=void -e entry 2>&1 | \
// RUN: FileCheck %s

// RUN: ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0:${ASAN_OPTIONS} \
// RUN: tpp-run %s -gpu=vulkan -print -seed 123 -gpu-wmma \
// RUN:  -entry-point-result=void -e entry 2>&1 | \
// RUN: FileCheck %s

func.func @entry(%arg0: memref<32x32xf16>, %arg1: memref<32x32xf16>, %arg2: memref<32x32xf16>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x2x16x16xf16>
    %expand_shape = memref.expand_shape %arg0 [[0, 1], [2, 3]] : memref<32x32xf16> into memref<2x16x2x16xf16>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<2x2x16x16xf16>
    linalg.transpose ins(%expand_shape : memref<2x16x2x16xf16>) outs(%alloc_0 : memref<2x2x16x16xf16>) permutation = [0, 2, 1, 3] 
    %expand_shape_1 = memref.expand_shape %arg1 [[0, 1], [2, 3]] : memref<32x32xf16> into memref<2x16x2x16xf16>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<2x2x16x16xf16>
    linalg.transpose ins(%expand_shape_1 : memref<2x16x2x16xf16>) outs(%alloc_2 : memref<2x2x16x16xf16>) permutation = [2, 0, 1, 3] 
    %expand_shape_3 = memref.expand_shape %arg2 [[0, 1], [2, 3]] : memref<32x32xf16> into memref<2x16x2x16xf16>
    linalg.transpose ins(%expand_shape_3 : memref<2x16x2x16xf16>) outs(%alloc : memref<2x2x16x16xf16>) permutation = [0, 2, 1, 3] 
    scf.forall (%arg3, %arg4) in (2, 2) {
      %subview = memref.subview %alloc_0[%arg3, 0, 0, 0] [1, 2, 16, 16] [1, 1, 1, 1] : memref<2x2x16x16xf16> to memref<2x16x16xf16, strided<[256, 16, 1], offset: ?>>
      %subview_5 = memref.subview %alloc_2[%arg4, 0, 0, 0] [1, 2, 16, 16] [1, 1, 1, 1] : memref<2x2x16x16xf16> to memref<2x16x16xf16, strided<[256, 16, 1], offset: ?>>
      %subview_6 = memref.subview %alloc[%arg3, %arg4, 0, 0] [1, 1, 16, 16] [1, 1, 1, 1] : memref<2x2x16x16xf16> to memref<16x16xf16, strided<[16, 1], offset: ?>>
      linalg.batch_reduce_matmul ins(%subview, %subview_5 : memref<2x16x16xf16, strided<[256, 16, 1], offset: ?>>, memref<2x16x16xf16, strided<[256, 16, 1], offset: ?>>) outs(%subview_6 : memref<16x16xf16, strided<[16, 1], offset: ?>>)
    }
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<2x16x2x16xf16>
    linalg.transpose ins(%alloc : memref<2x2x16x16xf16>) outs(%alloc_4 : memref<2x16x2x16xf16>) permutation = [0, 2, 1, 3] 
    %collapse_shape = memref.collapse_shape %alloc_4 [[0, 1], [2, 3]] : memref<2x16x2x16xf16> into memref<32x32xf16>
    linalg.copy ins(%collapse_shape : memref<32x32xf16>) outs(%arg2 : memref<32x32xf16>)

    return
  }

// CHECK: ( 0.036{{[0-9]+}}, 0.089{{[0-9]+}}, 0.086{{[0-9]+}}, 0.35{{[0-9]+}}
// CHECK: ( 0.075{{[0-9]+}}, 0.38{{[0-9]+}}, 0.35{{[0-9]+}}, 0.27{{[0-9]+}}
// CHECK: ( 0.008{{[0-9]+}}, 0.083{{[0-9]+}}, 0.18{{[0-9]+}}, 0.15{{[0-9]+}}
