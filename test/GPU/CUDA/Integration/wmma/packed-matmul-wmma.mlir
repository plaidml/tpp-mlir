// RUN: ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0:${ASAN_OPTIONS} \
// RUN: tpp-run %s -gpu=cuda -gpu-wmma \
// RUN:  -entry-point-result=void -e entry 2>&1 | \
// RUN: FileCheck %s

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

func.func @entry(%arg0: tensor<2x4x16x16xf16>, %arg1: tensor<4x4x16x16xf16>, %arg2: tensor<2x4x16x16xf16>) -> tensor<2x4x16x16xf16> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
    ins(%arg0, %arg1 : tensor<2x4x16x16xf16>, tensor<4x4x16x16xf16>) outs(%arg2 : tensor<2x4x16x16xf16>) {
    ^bb0(%in: f16, %in_2: f16, %out: f16):
      %4 = arith.mulf %in, %in_2 : f16
      %5 = arith.addf %out, %4 : f16
      linalg.yield %5 : f16
    } -> tensor<2x4x16x16xf16>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %vcst = arith.constant -1.000000e+00 : f16

  %buf = bufferization.to_memref %0 {read_only} : memref<2x4x16x16xf16>
  %out = memref.alloc() : memref<2x4x16x16xf16>
  %tOut = gpu.memcpy async %out, %buf : memref<2x4x16x16xf16>, memref<2x4x16x16xf16>
  gpu.wait [%tOut]
  %v0 = vector.transfer_read %out[%c1, %c2, %c0, %c0], %vcst : memref<2x4x16x16xf16>, vector<16x16xf16>
  vector.print %v0 : vector<16x16xf16>
  memref.dealloc %out : memref<2x4x16x16xf16>

  return %0 : tensor<2x4x16x16xf16>
}

// CHECK-COUNT-16: ( 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65 )
