// RUN: tpp-run %s -n 10 \
// RUN:  -e entry -entry-point-result=void

// Total flops = matmul O(2*n*m*k)
// 2*128x768x2304 = 452984832
// BENCH_TOTAL_FLOPS: 452984832

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

!A = tensor<2x36x64x64xf32>
!B = tensor<16x36x64x48xf32>
!C = tensor<2x16x64x48xf32>

// GEMM packed with tile size: 64, 48, 64
func.func @entry(%arg0: !A, %arg1: !B, %output: !C) -> !C {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
  ins(%arg0, %arg1 : !A, !B)
  outs(%output : !C) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %mul = arith.mulf %in, %in_0 : f32
      %add = arith.addf %out, %mul : f32
      linalg.yield %add : f32
  } -> !C

  return %0 : !C
}
