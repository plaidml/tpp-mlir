#map0 = affine_map<(d0, d1, d2, d3) -> (d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d6)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
module @predict_function {
  func.func @main(%arg0: tensor<256x224x224x3xf32>, %arg1: tensor<7x7x3x64xf32> {stdx.const}, %arg2: tensor<64xf32> {stdx.const}, %arg3: tensor<1x1x64x256xf32> {stdx.const}, %arg4: tensor<256xf32> {stdx.const}, %arg5: tensor<3x3x64x256xf32> {stdx.const}, %arg6: tensor<256xf32> {stdx.const}) -> tensor<256x56x56x256xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.pad %arg0 low[0, 2, 2, 0] high[0, 3, 3, 0] {
    ^bb0(%arg7: index, %arg8: index, %arg9: index, %arg10: index):
      tensor.yield %cst : f32
    } : tensor<256x224x224x3xf32> to tensor<256x229x229x3xf32>
    %1 = linalg.init_tensor [256, 112, 112, 64] : tensor<256x112x112x64xf32>
    %2 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<64xf32>) outs(%1 : tensor<256x112x112x64xf32>) {
    ^bb0(%arg7: f32, %arg8: f32):
      linalg.yield %arg7 : f32
    } -> tensor<256x112x112x64xf32>
    %3 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%0, %arg1 : tensor<256x229x229x3xf32>, tensor<7x7x3x64xf32>) outs(%2 : tensor<256x112x112x64xf32>) attrs =  {iterator_ranges = [256, 112, 112, 64, 7, 7, 3]} {
    ^bb0(%arg7: f32, %arg8: f32, %arg9: f32):
      %15 = arith.mulf %arg7, %arg8 : f32
      %16 = arith.addf %arg9, %15 : f32
      linalg.yield %16 : f32
    } -> tensor<256x112x112x64xf32>
    %4 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3 : tensor<256x112x112x64xf32>) outs(%1 : tensor<256x112x112x64xf32>) {
    ^bb0(%arg7: f32, %arg8: f32):
      %15 = mathx.relu %arg7 : f32
      linalg.yield %15 : f32
    } -> tensor<256x112x112x64xf32>
    %5 = tensor.pad %4 low[0, 0, 0, 0] high[0, 1, 1, 0] {
    ^bb0(%arg7: index, %arg8: index, %arg9: index, %arg10: index):
      tensor.yield %cst : f32
    } : tensor<256x112x112x64xf32> to tensor<256x113x113x64xf32>
    %6 = linalg.init_tensor [256, 56, 56, 256] : tensor<256x56x56x256xf32>
    %7 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg4 : tensor<256xf32>) outs(%6 : tensor<256x56x56x256xf32>) {
    ^bb0(%arg7: f32, %arg8: f32):
      linalg.yield %arg7 : f32
    } -> tensor<256x56x56x256xf32>
    %8 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%5, %arg3 : tensor<256x113x113x64xf32>, tensor<1x1x64x256xf32>) outs(%7 : tensor<256x56x56x256xf32>) attrs =  {iterator_ranges = [256, 56, 56, 256, 1, 1, 64]} {
    ^bb0(%arg7: f32, %arg8: f32, %arg9: f32):
      %15 = arith.mulf %arg7, %arg8 : f32
      %16 = arith.addf %arg9, %15 : f32
      linalg.yield %16 : f32
    } -> tensor<256x56x56x256xf32>
    %9 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%8 : tensor<256x56x56x256xf32>) outs(%6 : tensor<256x56x56x256xf32>) {
    ^bb0(%arg7: f32, %arg8: f32):
      %15 = mathx.relu %arg7 : f32
      linalg.yield %15 : f32
    } -> tensor<256x56x56x256xf32>
    %10 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg6 : tensor<256xf32>) outs(%6 : tensor<256x56x56x256xf32>) {
    ^bb0(%arg7: f32, %arg8: f32):
      linalg.yield %arg7 : f32
    } -> tensor<256x56x56x256xf32>
    %11 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%5, %arg5 : tensor<256x113x113x64xf32>, tensor<3x3x64x256xf32>) outs(%10 : tensor<256x56x56x256xf32>) attrs =  {iterator_ranges = [256, 56, 56, 256, 3, 3, 64]} {
    ^bb0(%arg7: f32, %arg8: f32, %arg9: f32):
      %15 = arith.mulf %arg7, %arg8 : f32
      %16 = arith.addf %arg9, %15 : f32
      linalg.yield %16 : f32
    } -> tensor<256x56x56x256xf32>
    %12 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%11 : tensor<256x56x56x256xf32>) outs(%6 : tensor<256x56x56x256xf32>) {
    ^bb0(%arg7: f32, %arg8: f32):
      %15 = mathx.relu %arg7 : f32
      linalg.yield %15 : f32
    } -> tensor<256x56x56x256xf32>
    %13 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%9, %12 : tensor<256x56x56x256xf32>, tensor<256x56x56x256xf32>) outs(%6 : tensor<256x56x56x256xf32>) {
    ^bb0(%arg7: f32, %arg8: f32, %arg9: f32):
      %15 = arith.addf %arg7, %arg8 : f32
      linalg.yield %15 : f32
    } -> tensor<256x56x56x256xf32>
    %14 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%13 : tensor<256x56x56x256xf32>) outs(%6 : tensor<256x56x56x256xf32>) {
    ^bb0(%arg7: f32, %arg8: f32):
      %15 = mathx.relu %arg7 : f32
      linalg.yield %15 : f32
    } -> tensor<256x56x56x256xf32>
    return %14 : tensor<256x56x56x256xf32>
  }
}
