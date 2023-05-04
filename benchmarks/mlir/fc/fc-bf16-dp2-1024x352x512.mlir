// RUN: tpp-run %s -n 10 \
// RUN:  -e entry -entry-point-result=void

// Total flops = matmul O(2*n*m*k) + BiasAdd O(n*m) + ReLU O(n*m)
// 2*1024x352x512 + 1024x352 + 1024x352 = 369819648
// BENCH_TOTAL_FLOPS: 369819648

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6 floordiv 2, d5, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!A = tensor<32x16x32x32xbf16>
!B = tensor<11x16x16x32x2xbf16>
!C = tensor<32x11x32x32xbf16>

// GEMM packed with tile size: 32, 32, 32
func.func @entry(%arg0: !A, %arg1: !B, %bias: !C, %output: !C) -> !C {
  %cst = arith.constant 0.000000e+00 : bf16
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]}
  ins(%arg0, %arg1 : !A, !B)
  outs(%output : !C) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %mul = arith.mulf %in, %in_0 : bf16
      %add = arith.addf %out, %mul : bf16
      linalg.yield %add : bf16
  } -> !C
  %1 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
  ins(%bias : !C)
  outs(%0 : !C) {
    ^bb0(%in: bf16, %out: bf16):
      %add = arith.addf %in, %out : bf16
      linalg.yield %add : bf16
  } -> !C
  %2 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
  outs(%1 : !C) {
    ^bb0(%out: bf16):
      %max = arith.maxf %out, %cst : bf16
      linalg.yield %max : bf16
  } -> !C

  return %2 : !C
}
