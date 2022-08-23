// RUN: standalone-opt %s -collapse-adjacent-dims -split-input-file | FileCheck %s

//#map0 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d5, d2, d3, d6)>
//#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d5, d6, d4)>
//#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4)>
//module {
//  func.func @collapse(%arg0: tensor<14x16x28x28x32xf32>, %arg1: tensor<32x16x1x1x32x32xf32>, %arg2: tensor<14x32x28x28x32xf32>) -> tensor<14x32x28x28x32xf32> {
//    %0 = tensor.collapse_shape %arg1 [[0], [1, 2, 3], [4], [5]] : tensor<32x16x1x1x32x32xf32> into tensor<32x16x32x32xf32>
//    %1 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg0, %0 : tensor<14x16x28x28x32xf32>, tensor<32x16x32x32xf32>) outs(%arg2 : tensor<14x32x28x28x32xf32>) {
//    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
//      %2 = arith.mulf %arg3, %arg4 : f32
//      %3 = arith.addf %arg5, %2 : f32
//      linalg.yield %3 : f32
//    } -> tensor<14x32x28x28x32xf32>
//    return %1 : tensor<14x32x28x28x32xf32>
//  }
//}

// -----

//#map0 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d5, 0, 0, d6)>
//#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d5, d6, d4)>
//#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4)>
//module {
//  func.func @collapse(%arg0: tensor<14x16x1x1x32xf32>, %arg1: tensor<32x16x1x1x32x32xf32>, %arg2: tensor<14x32x1x1x32xf32>) -> tensor<14x32x1x1x32xf32> {
//    %0 = tensor.collapse_shape %arg1 [[0], [1, 2, 3], [4], [5]] : tensor<32x16x1x1x32x32xf32> into tensor<32x16x32x32xf32>
//    %1 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg0, %0 : tensor<14x16x1x1x32xf32>, tensor<32x16x32x32xf32>) outs(%arg2 : tensor<14x32x1x1x32xf32>) {
//    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
//      %2 = arith.mulf %arg3, %arg4 : f32
//      %3 = arith.addf %arg5, %2 : f32
//      linalg.yield %3 : f32
//    } -> tensor<14x32x1x1x32xf32>
//    return %1 : tensor<14x32x1x1x32xf32>
//  }
//}

// -----

#map0 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 32 + d4, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 32 + d5, d1 * 32 + d4, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
#map6 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
module {
  func.func @conv(%arg0: tensor<14x512x28x28xf32>, %arg1: tensor<1024x512x1x1xf32>, %arg2: tensor<14x1024x28x28xf32>) -> tensor<14x1024x28x28xf32> {
    %0 = linalg.init_tensor [14, 16, 28, 28, 32] : tensor<14x16x28x28x32xf32>
    %1 = linalgx.relayout ins(%arg0 : tensor<14x512x28x28xf32>, #map0) outs(%0 : tensor<14x16x28x28x32xf32>, #map1) -> tensor<14x16x28x28x32xf32>
    %2 = linalg.init_tensor [32, 16, 1, 1, 32, 32] : tensor<32x16x1x1x32x32xf32>
    %3 = linalgx.relayout ins(%arg1 : tensor<1024x512x1x1xf32>, #map2) outs(%2 : tensor<32x16x1x1x32x32xf32>, #map3) -> tensor<32x16x1x1x32x32xf32>
    %4 = linalg.init_tensor [14, 32, 28, 28, 32] : tensor<14x32x28x28x32xf32>
    %5 = linalgx.relayout ins(%arg2 : tensor<14x1024x28x28xf32>, #map0) outs(%4 : tensor<14x32x28x28x32xf32>, #map1) -> tensor<14x32x28x28x32xf32>
    %6 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"], library_call = "tpp.BlockedConv2DNchwFchwOp"} ins(%1, %3 : tensor<14x16x28x28x32xf32>, tensor<32x16x1x1x32x32xf32>) outs(%5 : tensor<14x32x28x28x32xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %8 = arith.mulf %arg3, %arg4 : f32
      %9 = arith.addf %arg5, %8 : f32
      linalg.yield %9 : f32
    } -> tensor<14x32x28x28x32xf32>
    %7 = linalgx.relayout ins(%6 : tensor<14x32x28x28x32xf32>, #map1) outs(%arg2 : tensor<14x1024x28x28xf32>, #map0) -> tensor<14x1024x28x28xf32>
    return %7 : tensor<14x1024x28x28xf32>
  }
}
