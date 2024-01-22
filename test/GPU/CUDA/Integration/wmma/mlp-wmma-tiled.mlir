// RUN: ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0:${ASAN_OPTIONS} \
// RUN: tpp-run %s -gpu=cuda -gpu-wmma -print \
// RUN:  -entry-point-result=void -e entry 2>&1 | \
// RUN: FileCheck %s

// XFAIL:*
// See: #870

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @entry(%arg0: memref<32x32xf16>, %arg1: memref<32x32xf16>, %arg2: memref<32x32xf16>, %arg3: memref<32x32xf16>) -> memref<32x32xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f16
  linalg.matmul ins(%arg0, %arg1 : memref<32x32xf16>, memref<32x32xf16>) outs(%arg3 : memref<32x32xf16>)
  linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg2 : memref<32x32xf16>) outs(%arg3 : memref<32x32xf16>) {
  ^bb0(%in: f16, %out: f16):
    %0 = arith.addf %in, %out : f16
    linalg.yield %0 : f16
  }
  linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%arg3 :memref<32x32xf16>) {
  ^bb0(%out: f16):
    %0 = arith.maximumf %out, %cst : f16
    linalg.yield %0 : f16
  }
  return %arg3 : memref<32x32xf16>
}

// CHECK-COUNT-32: ( 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34 )
