// RUN: standalone-opt %s | standalone-opt | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>
module @get_value  {
  // CHECK-LABEL: func.func @main
  func.func @main(%arg0: tensor<2048x1000xf32>) -> tensor<2048x1000xf32> {
    %cst = arith.constant -0.0443678237 : f32
    %cst_0 = arith.constant 0.0887356474 : f32
    %1 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0, %cst_0 : tensor<2048x1000xf32>, f32) outs(%arg0 : tensor<2048x1000xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):  // no predecessors
      %3 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %3 : f32
    } -> tensor<2048x1000xf32>
    %2 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]} ins(%1, %cst : tensor<2048x1000xf32>, f32) outs(%arg0 : tensor<2048x1000xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):  // no predecessors
      %3 = arith.addf %arg1, %arg2 : f32
      linalg.yield %3 : f32
    } -> tensor<2048x1000xf32>
    return %2 : tensor<2048x1000xf32>
  }
}
