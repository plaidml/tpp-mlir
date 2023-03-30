// RUN: tpp-run %s -n 10 \
// RUN:  -e entry -entry-point-result=void

// Total flops = matmul O(2*n*m*k) x 3
// 2*256x1024x1024 (536870912) x 3 = 1,610,612,736
// BENCH_TOTAL_FLOPS: 1610612736

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

func.func @entry(%arg0: tensor<8x32x32x32xf32>, %arg3: tensor<8x32x32x32xf32>, %arg6: tensor<8x32x32x32xf32>, %arg9: tensor<8x32x32x32xf32> ) -> tensor<8x32x32x32xf32> {
  %arg1 = arith.constant dense<0.01> : tensor<32x32x32x32xf32>
  %arg4 = arith.constant dense<0.02> : tensor<32x32x32x32xf32>
  %arg7 = arith.constant dense<0.03> : tensor<32x32x32x32xf32>
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<8x32x32x32xf32>, tensor<32x32x32x32xf32>) outs(%arg3 : tensor<8x32x32x32xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %mul = arith.mulf %in, %in_0 : f32
      %add = arith.addf %out, %mul : f32
      linalg.yield %add : f32
  } -> tensor<8x32x32x32xf32>

  %4 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%0, %arg4 : tensor<8x32x32x32xf32>, tensor<32x32x32x32xf32>) outs(%arg6 : tensor<8x32x32x32xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %mul = arith.mulf %in, %in_0 : f32
      %add = arith.addf %out, %mul : f32
      linalg.yield %add : f32
  } -> tensor<8x32x32x32xf32>

  %8 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%4, %arg7 : tensor<8x32x32x32xf32>, tensor<32x32x32x32xf32>) outs(%arg9 : tensor<8x32x32x32xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %mul = arith.mulf %in, %in_0 : f32
      %add = arith.addf %out, %mul : f32
      linalg.yield %add : f32
  } -> tensor<8x32x32x32xf32>

  return %8 : tensor<8x32x32x32xf32>
}

