// RUN: tpp-run %s -n 1000 \
// RUN:  -e entry -entry-point-result=void

// Total flops = matmul O(2*n*m*k) + BiasAdd (n*m) + ReLU (O(n*m) x 3
// 2*256x1024x1024 (536870912) + 256x1024 (262144) + 256x1024 262144) x 3 = 1,072,168,960
// BENCH_TOTAL_FLOPS: 1072168960

#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @entry(%arg0: tensor<256x512xf32>, %arg3: tensor<256x512xf32>, %arg6: tensor<256x512xf32>, %arg9: tensor<256x512xf32>, %arg1: tensor<512x512xf32>, %arg4: tensor<512x512xf32>, %arg7: tensor<512x512xf32>) -> tensor<256x512xf32> {
  %0 = linalg.generic {
    indexing_maps = [#map2, #map3, #map4], 
    iterator_types = ["parallel", "parallel", "reduction"]} 
    ins(%arg0, %arg1 : tensor<256x512xf32>, tensor<512x512xf32>) 
    outs(%arg3 : tensor<256x512xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %mul = arith.mulf %in, %in_0 : f32
        %add = arith.addf %out, %mul : f32
        linalg.yield %add : f32
  } -> tensor<256x512xf32>
  %4 = linalg.generic {
    indexing_maps = [#map2, #map3, #map4],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%0, %arg4 : tensor<256x512xf32>, tensor<512x512xf32>) outs(%arg6 : tensor<256x512xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %mul = arith.mulf %in, %in_0 : f32
        %add = arith.addf %out, %mul : f32
        linalg.yield %add : f32
  } -> tensor<256x512xf32>
  %8 = linalg.generic {
    indexing_maps = [#map2, #map3, #map4], 
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%4, %arg7 : tensor<256x512xf32>, tensor<512x512xf32>) outs(%arg9 : tensor<256x512xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %mul = arith.mulf %in, %in_0 : f32
        %add = arith.addf %out, %mul : f32
        linalg.yield %add : f32
  } -> tensor<256x512xf32>
  return %8 : tensor<256x512xf32>
}
