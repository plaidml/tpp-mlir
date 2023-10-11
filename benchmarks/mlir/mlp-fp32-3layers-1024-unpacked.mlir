// RUN: tpp-run %s -n 10 \
// RUN:  -e entry -entry-point-result=void

// Total flops = matmul O(2*n*m*k) + BiasAdd (n*m) + ReLU (O(n*m) x 3
// ( 2*256x1024x1024 (536870912) + 256x1024 (262144) + 256x1024 (262144) ) x 3 = 1,612,185,600
// BENCH_TOTAL_FLOPS: 1612185600

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>

func.func @entry(%arg0: tensor<256x1024xf32>, %arg3: tensor<256x1024xf32>, %arg6: tensor<256x1024xf32>, %arg9: tensor<256x1024xf32> ) -> tensor<256x1024xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  
  %arg1 = arith.constant dense<1.0> : tensor<1024x1024xf32>
  %arg4 = arith.constant dense<1.0> : tensor<1024x1024xf32>
  %arg7 = arith.constant dense<1.0> : tensor<1024x1024xf32>
  %arg2 = arith.constant dense<1.0> : tensor<256x1024xf32>
  %arg5 = arith.constant dense<1.0> : tensor<256x1024xf32>
  %arg8 = arith.constant dense<1.0> : tensor<256x1024xf32>
  
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<256x1024xf32>, tensor<1024x1024xf32>) 
                     outs(%arg3 : tensor<256x1024xf32>) -> tensor<256x1024xf32>
  %2 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]}
  ins(%0, %arg2 : tensor<256x1024xf32>, tensor<256x1024xf32>) outs(%arg3 : tensor<256x1024xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %add = arith.addf %in, %in_0 : f32
      linalg.yield %add : f32
  } -> tensor<256x1024xf32>
  %3 = linalg.generic {
    indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} 
    ins(%2 : tensor<256x1024xf32>) outs(%arg3 : tensor<256x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %max = arith.maximumf %in, %cst : f32
      linalg.yield %max : f32
  } -> tensor<256x1024xf32>

  %4 = linalg.matmul ins(%3, %arg4 : tensor<256x1024xf32>, tensor<1024x1024xf32>) 
    outs(%arg6 : tensor<256x1024xf32>) -> tensor<256x1024xf32>
  %6 = linalg.generic {
    indexing_maps = [#map3, #map3, #map3], 
    iterator_types = ["parallel", "parallel"]} 
    ins(%4, %arg5 : tensor<256x1024xf32>, tensor<256x1024xf32>) outs(%arg6 : tensor<256x1024xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %add = arith.addf %in, %in_0 : f32
      linalg.yield %add : f32
  } -> tensor<256x1024xf32>
  %7 = linalg.generic {
    indexing_maps = [#map3, #map3], 
    iterator_types = ["parallel", "parallel"]} 
    ins(%6 : tensor<256x1024xf32>) outs(%arg6 : tensor<256x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %max = arith.maximumf %in, %cst : f32
      linalg.yield %max : f32
  } -> tensor<256x1024xf32>

  %8 = linalg.matmul ins(%7, %arg7 : tensor<256x1024xf32>, tensor<1024x1024xf32>) 
                     outs(%arg9 : tensor<256x1024xf32>) -> tensor<256x1024xf32>
  %10 = linalg.generic {
    indexing_maps = [#map3, #map3, #map3], 
    iterator_types = ["parallel", "parallel"]} 
    ins(%8, %arg8 : tensor<256x1024xf32>, tensor<256x1024xf32>) outs(%arg9 : tensor<256x1024xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %add = arith.addf %in, %in_0 : f32
      linalg.yield %add : f32
  } -> tensor<256x1024xf32>
  %11 = linalg.generic {
    indexing_maps = [#map3, #map3], 
    iterator_types = ["parallel", "parallel"]} 
    ins(%10 : tensor<256x1024xf32>) outs(%arg9 : tensor<256x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %max = arith.maximumf %in, %cst : f32
      linalg.yield %max : f32
  } -> tensor<256x1024xf32>

  return %11 : tensor<256x1024xf32>
}
