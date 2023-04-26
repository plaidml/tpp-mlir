// RUN: tpp-run %s -n 10 \
// RUN:  -e entry -entry-point-result=void

// Total flops = matmul O(2*n*m*k)
// 2*256x1024x1024 = 536870912
// BENCH_TOTAL_FLOPS: 536870912

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6 floordiv 2, d5, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>

!A = tensor<4x16x64x64xbf16>
!B = tensor<16x16x32x64x2xbf16>
!C = tensor<4x16x64x64xbf16>

// GEMM packed with tile size: 64, 64, 64
func.func @entry(%arg0: !A, %arg1: !B, %output: !C) -> !C {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]}
  ins(%arg0, %arg1 : !A, !B)
  outs(%output : !C) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %mul = arith.mulf %in, %in_0 : bf16
      %add = arith.addf %out, %mul : bf16
      linalg.yield %add : bf16
  } -> !C

  return %0 : !C
}
