// RUN: tpp-run %s -n 10 \
// RUN:  -e entry -entry-point-result=void

// Total flops = matmul O(2*n*m*k)
// 2*256x1024x1024 = 536870912
// BENCH_TOTAL_FLOPS: 536870912

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

// GEMM 256 x 1024 x 1024 packed into 8x32 (256) x 32x32 (1024)
func.func @entry(%arg0: tensor<8x32x32x32xbf16>, %arg1: tensor<32x32x32x32xbf16>, %output: tensor<8x32x32x32xbf16>) -> tensor<8x32x32x32xbf16> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<8x32x32x32xbf16>, tensor<32x32x32x32xbf16>) outs(%output : tensor<8x32x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %mul = arith.mulf %in, %in_0 : bf16
      %add = arith.addf %out, %mul : bf16
      linalg.yield %add : bf16
  } -> tensor<8x32x32x32xbf16>

  return %0 : tensor<8x32x32x32xbf16>
}

