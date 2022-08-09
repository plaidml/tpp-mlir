// RUN: standalone-opt %s -map-linalg-to-tpp -to-block-layout="block-factor=32" -pre-bufferization -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize | FileCheck %s

// XFAIL: *
// TODO: linalg.generic -> linalg.matmul
#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
module @predict_function  {
  func.func @main(%arg0: tensor<128x256xf32>, 
                  %arg1: tensor<256x512xf32> {stdx.const}, 
                  %arg2: tensor<512xf32> {stdx.const}, 
                  %arg3: tensor<512x1024xf32> {stdx.const}, 
                  %arg4: tensor<1024xf32> {stdx.const}, 
                  %arg5: tensor<1024x2048xf32> {stdx.const}, 
                  %arg6: tensor<2048xf32> {stdx.const},
                  %arg7: tensor<2048x1024xf32> {stdx.const}, 
                  %arg8: tensor<1024xf32> {stdx.const}, 
                  %output: tensor<128x1024xf32> {stdx.res},
                  %output1: tensor<128x2048xf32> {stdx.const}, 
                  %output2: tensor<128x1024xf32> {stdx.const}, 
                  %ouput3: tensor<128x512xf32> {stdx.const}) -> tensor<128x1024xf32> {
    %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg2 : tensor<512xf32>) outs(%ouput3 : tensor<128x512xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      linalg.yield %arg9 : f32
    } -> tensor<128x512xf32>
    %2 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<128x256xf32>, tensor<256x512xf32>) outs(%1 : tensor<128x512xf32>) attrs =  {iterator_ranges = [128, 512, 256]} {
    ^bb0(%arg9: f32, %arg10: f32, %arg11: f32):  
      %16 = arith.mulf %arg9, %arg10 : f32
      %17 = arith.addf %arg11, %16 : f32
      linalg.yield %17 : f32
    } -> tensor<128x512xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<128x512xf32>) outs(%ouput3 : tensor<128x512xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):  
      %16 = mathx.relu %arg9 : f32
      linalg.yield %16 : f32
    } -> tensor<128x512xf32>
    %5 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg4 : tensor<1024xf32>) outs(%output2 : tensor<128x1024xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):  
      linalg.yield %arg9 : f32
    } -> tensor<128x1024xf32>
    %6 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %arg3 : tensor<128x512xf32>, tensor<512x1024xf32>) outs(%5 : tensor<128x1024xf32>) attrs =  {iterator_ranges = [128, 1024, 512]} {
    ^bb0(%arg9: f32, %arg10: f32, %arg11: f32):  
      %16 = arith.mulf %arg9, %arg10 : f32
      %17 = arith.addf %arg11, %16 : f32
      linalg.yield %17 : f32
    } -> tensor<128x1024xf32>
    %7 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<128x1024xf32>) outs(%output2 : tensor<128x1024xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):  
      %16 = mathx.relu %arg9 : f32 
      linalg.yield %16 : f32
    } -> tensor<128x1024xf32>
    return %7 : tensor<128x1024xf32>
  }
}
