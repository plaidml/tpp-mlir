// RUN: ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0:${ASAN_OPTIONS} \
// RUN: tpp-run %s -gpu=cuda -gpu-wmma -print \
// RUN:  -entry-point-result=void -e entry 2>&1 | \
// RUN: FileCheck %s

func.func @entry(%arg0: memref<16x32x32xf16>,
                 %arg1: memref<16x32x32xf16>,
                 %arg2: memref<32x32xf16>) -> memref<32x32xf16> {
  linalg.batch_reduce_matmul  ins(%arg0, %arg1 : memref<16x32x32xf16>, memref<16x32x32xf16>)
                              outs(%arg2 : memref<32x32xf16>)
  return %arg2 : memref<32x32xf16>
}

// CHECK-COUNT-32: ( 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513, 513 )
