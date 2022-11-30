#map0 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3, d0 * 32 + d4, d1 * 32 + d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d6)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
#map6 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
#map7 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d8 * 32 + d6)>
#map8 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d8, d7, d4, d5, d6, d3)>
#map9 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d7 * 32 + d3)>
module @predict_function  {
  func.func @main(%arg0: tensor<7x7x3x64xf32> {stdx.const}, %arg1: tensor<64xf32> {stdx.const}, %arg2: tensor<1x1x64x256xf32> {stdx.const}, %arg3: tensor<256xf32> {stdx.const}, %arg4: tensor<3x3x64x256xf32> {stdx.const}, %arg5: tensor<256xf32> {stdx.const}) {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = linalg.init_tensor [1, 112, 112, 64] : tensor<1x112x112x64xf32>
    %1 = linalg.init_tensor [1, 56, 56, 256] : tensor<1x56x56x256xf32>
    %2 = linalg.init_tensor [2, 8, 1, 1, 32, 32] : tensor<2x8x1x1x32x32xf32>
    %3 = linalgx.copy(%arg2, %2) {inputMap = #map0, outputMap = #map1} : tensor<1x1x64x256xf32>, tensor<2x8x1x1x32x32xf32> -> tensor<2x8x1x1x32x32xf32>
    %4 = linalg.init_tensor [2, 8, 3, 3, 32, 32] : tensor<2x8x3x3x32x32xf32>
    %5 = linalgx.copy(%arg4, %4) {inputMap = #map0, outputMap = #map1} : tensor<3x3x64x256xf32>, tensor<2x8x3x3x32x32xf32> -> tensor<2x8x3x3x32x32xf32>
    stdx.closure(%arg6: tensor<1x224x224x3xf32>) -> tensor<1x56x56x256xf32> {
      %6 = linalg.pad_tensor %arg6 low[0, 2, 2, 0] high[0, 3, 3, 0]  {
      ^bb0(%arg7: index, %arg8: index, %arg9: index, %arg10: index):  // no predecessors
        linalg.yield %cst : f32
      } : tensor<1x224x224x3xf32> to tensor<1x229x229x3xf32>
      %7 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<64xf32>) outs(%0 : tensor<1x112x112x64xf32>) {
      ^bb0(%arg7: f32, %arg8: f32):  // no predecessors
        linalg.yield %arg7 : f32
      } -> tensor<1x112x112x64xf32>
      %8 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%6, %arg0 : tensor<1x229x229x3xf32>, tensor<7x7x3x64xf32>) outs(%7 : tensor<1x112x112x64xf32>) attrs =  {iterator_ranges = [1, 112, 112, 64, 7, 7, 3]} {
      ^bb0(%arg7: f32, %arg8: f32, %arg9: f32):  // no predecessors
        %19 = arith.mulf %arg7, %arg8 : f32
        %20 = arith.addf %arg9, %19 : f32
        linalg.yield %20 : f32
      } -> tensor<1x112x112x64xf32>
      %9 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%8 : tensor<1x112x112x64xf32>) outs(%0 : tensor<1x112x112x64xf32>) {
      ^bb0(%arg7: f32, %arg8: f32):  // no predecessors
        %19 = stdx.relu(%arg7) : (f32) -> f32
        linalg.yield %19 : f32
      } -> tensor<1x112x112x64xf32>
      %10 = linalg.pad_tensor %9 low[0, 0, 0, 0] high[0, 1, 1, 0]  {
      ^bb0(%arg7: index, %arg8: index, %arg9: index, %arg10: index):  // no predecessors
        linalg.yield %cst : f32
      } : tensor<1x112x112x64xf32> to tensor<1x113x113x64xf32>
      %11 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg3 : tensor<256xf32>) outs(%1 : tensor<1x56x56x256xf32>) {
      ^bb0(%arg7: f32, %arg8: f32):  // no predecessors
        linalg.yield %arg7 : f32
      } -> tensor<1x56x56x256xf32>
      %12 = linalg.generic {indexing_maps = [#map7, #map8, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "parallel", "reduction"]} ins(%10, %3 : tensor<1x113x113x64xf32>, tensor<2x8x1x1x32x32xf32>) outs(%11 : tensor<1x56x56x256xf32>) {
      ^bb0(%arg7: f32, %arg8: f32, %arg9: f32):  // no predecessors
        %19 = arith.mulf %arg7, %arg8 : f32
        %20 = arith.addf %arg9, %19 : f32
        linalg.yield %20 : f32
      } -> tensor<1x56x56x256xf32>
      %13 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%12 : tensor<1x56x56x256xf32>) outs(%1 : tensor<1x56x56x256xf32>) {
      ^bb0(%arg7: f32, %arg8: f32):  // no predecessors
        %19 = stdx.relu(%arg7) : (f32) -> f32
        linalg.yield %19 : f32
      } -> tensor<1x56x56x256xf32>
      %14 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg5 : tensor<256xf32>) outs(%1 : tensor<1x56x56x256xf32>) {
      ^bb0(%arg7: f32, %arg8: f32):  // no predecessors
        linalg.yield %arg7 : f32
      } -> tensor<1x56x56x256xf32>
      %15 = linalg.generic {indexing_maps = [#map7, #map8, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "parallel", "reduction"]} ins(%10, %5 : tensor<1x113x113x64xf32>, tensor<2x8x3x3x32x32xf32>) outs(%14 : tensor<1x56x56x256xf32>) {
      ^bb0(%arg7: f32, %arg8: f32, %arg9: f32):  // no predecessors
        %19 = arith.mulf %arg7, %arg8 : f32
        %20 = arith.addf %arg9, %19 : f32
        linalg.yield %20 : f32
      } -> tensor<1x56x56x256xf32>
      %16 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%15 : tensor<1x56x56x256xf32>) outs(%1 : tensor<1x56x56x256xf32>) {
      ^bb0(%arg7: f32, %arg8: f32):  // no predecessors
        %19 = stdx.relu(%arg7) : (f32) -> f32
        linalg.yield %19 : f32
      } -> tensor<1x56x56x256xf32>
      %17 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%13, %16 : tensor<1x56x56x256xf32>, tensor<1x56x56x256xf32>) outs(%1 : tensor<1x56x56x256xf32>) {
      ^bb0(%arg7: f32, %arg8: f32, %arg9: f32):  // no predecessors
        %19 = arith.addf %arg7, %arg8 : f32
        linalg.yield %19 : f32
      } -> tensor<1x56x56x256xf32>
      %18 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%17 : tensor<1x56x56x256xf32>) outs(%1 : tensor<1x56x56x256xf32>) {
      ^bb0(%arg7: f32, %arg8: f32):  // no predecessors
        %19 = stdx.relu(%arg7) : (f32) -> f32
        linalg.yield %19 : f32
      } -> tensor<1x56x56x256xf32>
      stdx.yield %18 : tensor<1x56x56x256xf32>
    }
    return
  }
}
