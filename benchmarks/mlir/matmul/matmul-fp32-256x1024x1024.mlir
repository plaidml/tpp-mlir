// RUN: tpp-run %s -n 10 \
// RUN:  -e entry -entry-point-result=void

// Total flops = matmul O(2*n*m*k)
// 2*256x1024x1024 = 536870912
// BENCH_TOTAL_FLOPS: 536870912

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

// GEMM 256 x 1024 x 1024 packed into 8x32 (256) x 32x32 (1024)
func.func @entry(%arg0: tensor<8x32x32x32xf32>, %arg1: tensor<32x32x32x32xf32>, %output: tensor<8x32x32x32xf32>) -> tensor<8x32x32x32xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<8x32x32x32xf32>, tensor<32x32x32x32xf32>) outs(%output : tensor<8x32x32x32xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %mul = arith.mulf %in, %in_0 : f32
      %add = arith.addf %out, %mul : f32
      linalg.yield %add : f32
  } -> tensor<8x32x32x32xf32>

  return %0 : tensor<8x32x32x32xf32>
}

