// RUN: standalone-opt %s -linalg-bufferize -func-bufferize -canonicalize -cse -map-linalg-to-tpp | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
module @predict_function {
  // CHECK-LABEL: func.func @main
  func.func @main(%arg0: tensor<256x512xf32> {stdx.const}, %arg1: tensor<512xf32> {stdx.const}, %arg10: tensor<1x256xf32>) -> tensor<1x512xf32> {
    %0 = linalg.init_tensor [1, 512] : tensor<1x512xf32>
    // CHECK: tpp.identity
    %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<512xf32>) outs(%0 : tensor<1x512xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
    } -> tensor<1x512xf32>
    // CHECK: tpp.matmul
    %2 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg10, %arg0 : tensor<1x256xf32>, tensor<256x512xf32>) outs(%1 : tensor<1x512xf32>) attrs =  {iterator_ranges = [1, 512, 256]} {
      ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
        %4 = arith.mulf %arg3, %arg4 : f32
        %5 = arith.addf %arg5, %4 : f32
        linalg.yield %5 : f32
    } -> tensor<1x512xf32>
    // CHECK: tpp.relu
    %3 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<1x512xf32>) outs(%0 : tensor<1x512xf32>) {
      ^bb0(%arg3: f32, %arg4: f32):
        %4 = mathx.relu %arg3 : f32
        linalg.yield %4 : f32
      } -> tensor<1x512xf32>
    return %3 : tensor<1x512xf32>
  }
}
