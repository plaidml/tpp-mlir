// RUN: ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0:${ASAN_OPTIONS} \
// RUN: tpp-run %s -gpu=cuda -gpu-wmma -print \
// RUN:  -entry-point-result=void -e entry 2>&1 | \
// RUN: FileCheck %s

func.func @entry(%arg0: memref<16x32x16xf16>, %arg1: memref<16x64x16xf16>, %arg2: memref<32x32xf16>) -> memref<32x32xf16> {
  %subview = memref.subview %arg0[0, 0, 0] [16, 1, 16] [1, 1, 1]
    : memref<16x32x16xf16> to memref<16x16xf16, strided<[512, 1], offset: 0>>
  %subview_0 = memref.subview %arg1[0, 0, 0] [16, 1, 16] [1, 1, 1]
    : memref<16x64x16xf16> to memref<16x16xf16, strided<[1024, 1], offset: 0>>
  %subview_1 = memref.subview %arg2[16, 0] [16, 16] [1, 1]
    : memref<32x32xf16> to memref<16x16xf16, strided<[32, 1], offset: 512>>

  %c2 = arith.constant 2.0 : f16
  %c4 = arith.constant 4.0 : f16

  linalg.fill ins(%c2 : f16) outs(%subview : memref<16x16xf16, strided<[512, 1], offset: 0>>)
  linalg.fill ins(%c4 : f16) outs(%subview_0 : memref<16x16xf16, strided<[1024, 1], offset: 0>>)

  linalg.matmul ins(%subview, %subview_0 : memref<16x16xf16, strided<[512, 1], offset: 0>>,
                                           memref<16x16xf16, strided<[1024, 1], offset: 0>>)
                outs(%subview_1 : memref<16x16xf16, strided<[32, 1], offset: 512>>)
  return %arg2 : memref<32x32xf16>
}

// CHECK-COUNT-16: ( 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 )
// CHECK-COUNT-16: ( 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 129, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 )
