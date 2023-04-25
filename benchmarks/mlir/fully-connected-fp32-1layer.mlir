// RUN: tpp-opt %s -tile-consumer-and-fuse-producers -convert-linalg-to-tpp -bufferize |
// RUN: tpp-run %s -n 10 \
// RUN:  -e entry -entry-point-result=void

// Total flops = matmul O(2*n*m*k) = 2 * 512 * 512 * 256 2 * 256 * 512 = 134,479,872
// BENCH_TOTAL_FLOPS: 134479872

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>

func.func @entry(%arg0: tensor<256x512xf32>, %arg1: tensor<512x512xf32>) -> tensor<256x512xf32> {
  %cst = arith.constant dense<0.00999999977> : tensor<256x512xf32>
  %cst_2 = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<256x512xf32>
  %1 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<256x512xf32>) -> tensor<256x512xf32>
  %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<256x512xf32>, tensor<512x512xf32>) outs(%1 : tensor<256x512xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %5 = arith.mulf %in, %in_3 : f32
      %6 = arith.addf %out, %5 : f32
      linalg.yield %6 : f32
  } -> tensor<256x512xf32>
  %3 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%2, %cst : tensor<256x512xf32>, tensor<256x512xf32>) outs(%1 : tensor<256x512xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %5 = arith.addf %in, %in_3 : f32
      linalg.yield %5 : f32
  } -> tensor<256x512xf32>
  %4 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<256x512xf32>) outs(%1 : tensor<256x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %5 = arith.maxf %in, %cst_2 : f32
      linalg.yield %5 : f32
  } -> tensor<256x512xf32>
  return %4 : tensor<256x512xf32>
}
