// RUN: standalone-opt -transform-dialect-interpreter %s | FileCheck %s

#map0 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 32 + d4, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 32 + d5, d1 * 32 + d4, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d5, d2, d3, d6)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d5, d6, d4)>
#map6 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4)>
module {
  transform.with_pdl_patterns {
  ^bb0(%arg0: !pdl.operation):
    sequence %arg0 failures(propagate) {
    ^bb0(%arg1: !pdl.operation):
      %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
      %1 = transform.structured.collapsing %0 [[0], [1], [2, 3], [4], [5], [6]]
    }
  }
  // CHECK: conv
  func.func @conv(%arg0: tensor<14x512x28x28xf32>, %arg1: tensor<1024x512x1x1xf32>, %arg2: tensor<14x1024x28x28xf32>) -> tensor<14x1024x28x28xf32> {
    %0 = linalg.init_tensor [14, 16, 28, 28, 32] : tensor<14x16x28x28x32xf32>
    %1 = linalgx.relayout ins(%arg0 : tensor<14x512x28x28xf32>, #map0) outs(%0 : tensor<14x16x28x28x32xf32>, #map1) -> tensor<14x16x28x28x32xf32>
    %2 = linalg.init_tensor [32, 16, 1, 1, 32, 32] : tensor<32x16x1x1x32x32xf32>
    %3 = linalgx.relayout ins(%arg1 : tensor<1024x512x1x1xf32>, #map2) outs(%2 : tensor<32x16x1x1x32x32xf32>, #map3) -> tensor<32x16x1x1x32x32xf32>
    %4 = linalg.init_tensor [14, 32, 28, 28, 32] : tensor<14x32x28x28x32xf32>
    %5 = linalgx.relayout ins(%arg2 : tensor<14x1024x28x28xf32>, #map0) outs(%4 : tensor<14x32x28x28x32xf32>, #map1) -> tensor<14x32x28x28x32xf32>
    %6 = tensor.collapse_shape %3 [[0], [1, 2, 3], [4], [5]] : tensor<32x16x1x1x32x32xf32> into tensor<32x16x32x32xf32>
    %7 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%1, %6 : tensor<14x16x28x28x32xf32>, tensor<32x16x32x32xf32>) outs(%5 : tensor<14x32x28x28x32xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %9 = arith.mulf %arg3, %arg4 : f32
      %10 = arith.addf %arg5, %9 : f32
      linalg.yield %10 : f32
    } -> tensor<14x32x28x28x32xf32>
    %8 = linalgx.relayout ins(%7 : tensor<14x32x28x28x32xf32>, #map1) outs(%arg2 : tensor<14x1024x28x28xf32>, #map0) -> tensor<14x1024x28x28xf32>
    return %8 : tensor<14x1024x28x28xf32>
  }
}
