// RUN: tpp-opt %s -element-wise-fusion |
// RUN: tpp-run %s -n 1000 \
// RUN:  -e entry -entry-point-result=void

// Total flops = matmul O(2*n*m*k) x 3
// 2*256x1024x1024 (536870912) x 3 = 1,610,612,736
// BENCH_TOTAL_FLOPS: 1610612736

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6 floordiv 2, d5, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>

func.func @entry(%arg0: tensor<8x32x32x32xbf16>, %arg3: tensor<8x32x32x32xbf16>, %arg6: tensor<8x32x32x32xbf16>, %arg9: tensor<8x32x32x32xbf16> ) -> tensor<8x32x32x32xbf16> {
  %arg1 = arith.constant dense<0.01> : tensor<32x32x8x32x4xbf16>
  %arg4 = arith.constant dense<0.02> : tensor<32x32x8x32x4xbf16>
  %arg7 = arith.constant dense<0.03> : tensor<32x32x8x32x4xbf16>
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<8x32x32x32xbf16>, tensor<32x32x8x32x4xbf16>) outs(%arg3 : tensor<8x32x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %mul = arith.mulf %in, %in_0 : bf16
      %add = arith.addf %out, %mul : bf16
      linalg.yield %add : bf16
  } -> tensor<8x32x32x32xbf16>

  %4 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%0, %arg4 : tensor<8x32x32x32xbf16>, tensor<32x32x8x32x4xbf16>) outs(%arg6 : tensor<8x32x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %mul = arith.mulf %in, %in_0 : bf16
      %add = arith.addf %out, %mul : bf16
      linalg.yield %add : bf16
  } -> tensor<8x32x32x32xbf16>

  %8 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%4, %arg7 : tensor<8x32x32x32xbf16>, tensor<32x32x8x32x4xbf16>) outs(%arg9 : tensor<8x32x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %mul = arith.mulf %in, %in_0 : bf16
      %add = arith.addf %out, %mul : bf16
      linalg.yield %add : bf16
  } -> tensor<8x32x32x32xbf16>

  return %8 : tensor<8x32x32x32xbf16>
}

