#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
module @predict_function {
  func.func @main(%arg0: tensor<1x256xf32>, %arg1: tensor<256x512xf32> {stdx.const}, %arg2: tensor<512xf32> {stdx.const}, %arg3: tensor<512x1024xf32> {stdx.const}, %arg4: tensor<1024xf32> {stdx.const}, %arg5: tensor<1024x2048xf32> {stdx.const}, %arg6: tensor<2048xf32> {stdx.const}, %arg7: tensor<2048x1000xf32> {stdx.const}, %arg8: tensor<1000xf32> {stdx.const}) -> tensor<1x1000xf32> {
    %0 = linalg.init_tensor [1, 512] : tensor<1x512xf32>
    %1 = linalg.init_tensor [1, 1024] : tensor<1x1024xf32>
    %2 = linalg.init_tensor [1, 2048] : tensor<1x2048xf32>
    %3 = linalg.init_tensor [1, 1000] : tensor<1x1000xf32>
    %4 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg2 : tensor<512xf32>) outs(%0 : tensor<1x512xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      linalg.yield %arg9 : f32
    } -> tensor<1x512xf32>
    %5 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<1x256xf32>, tensor<256x512xf32>) outs(%4 : tensor<1x512xf32>) attrs =  {iterator_ranges = [1, 512, 256]} {
    ^bb0(%arg9: f32, %arg10: f32, %arg11: f32):
      %16 = arith.mulf %arg9, %arg10 : f32
      %17 = arith.addf %arg11, %16 : f32
      linalg.yield %17 : f32
    } -> tensor<1x512xf32>
    %6 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<1x512xf32>) outs(%0 : tensor<1x512xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      %16 = mathx.relu %arg9 : f32
      linalg.yield %16 : f32
    } -> tensor<1x512xf32>
    %7 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg4 : tensor<1024xf32>) outs(%1 : tensor<1x1024xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      linalg.yield %arg9 : f32
    } -> tensor<1x1024xf32>
    %8 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%6, %arg3 : tensor<1x512xf32>, tensor<512x1024xf32>) outs(%7 : tensor<1x1024xf32>) attrs =  {iterator_ranges = [1, 1024, 512]} {
    ^bb0(%arg9: f32, %arg10: f32, %arg11: f32):
      %16 = arith.mulf %arg9, %arg10 : f32
      %17 = arith.addf %arg11, %16 : f32
      linalg.yield %17 : f32
    } -> tensor<1x1024xf32>
    %9 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%8 : tensor<1x1024xf32>) outs(%1 : tensor<1x1024xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      %16 = mathx.relu %arg9 : f32
      linalg.yield %16 : f32
    } -> tensor<1x1024xf32>
    %10 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg6 : tensor<2048xf32>) outs(%2 : tensor<1x2048xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      linalg.yield %arg9 : f32
    } -> tensor<1x2048xf32>
    %11 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%9, %arg5 : tensor<1x1024xf32>, tensor<1024x2048xf32>) outs(%10 : tensor<1x2048xf32>) attrs =  {iterator_ranges = [1, 2048, 1024]} {
    ^bb0(%arg9: f32, %arg10: f32, %arg11: f32):
      %16 = arith.mulf %arg9, %arg10 : f32
      %17 = arith.addf %arg11, %16 : f32
      linalg.yield %17 : f32
    } -> tensor<1x2048xf32>
    %12 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%11 : tensor<1x2048xf32>) outs(%2 : tensor<1x2048xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      %16 = mathx.relu %arg9 : f32
      linalg.yield %16 : f32
    } -> tensor<1x2048xf32>
    %13 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg8 : tensor<1000xf32>) outs(%3 : tensor<1x1000xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      linalg.yield %arg9 : f32
    } -> tensor<1x1000xf32>
    %14 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%12, %arg7 : tensor<1x2048xf32>, tensor<2048x1000xf32>) outs(%13 : tensor<1x1000xf32>) attrs =  {iterator_ranges = [1, 1000, 2048]} {
    ^bb0(%arg9: f32, %arg10: f32, %arg11: f32):
      %16 = arith.mulf %arg9, %arg10 : f32
      %17 = arith.addf %arg11, %16 : f32
      linalg.yield %17 : f32
    } -> tensor<1x1000xf32>
    %15 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%14 : tensor<1x1000xf32>) outs(%3 : tensor<1x1000xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      %16 = mathx.relu %arg9 : f32
      linalg.yield %16 : f32
    } -> tensor<1x1000xf32>
    return %14 : tensor<1x1000xf32>
  }
}
