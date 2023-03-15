//RUN: tpp-opt %s -decompose-linalg-ops | FileCheck %s
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>

module {
  func.func @mlp(%arg0: tensor<4x4xbf16>, %arg1: tensor<4xbf16>, %arg2: tensor<4x4xbf16>) -> tensor<4x4xbf16> {
    %cst = arith.constant 0.0:bf16
    // CHECK: %[[temp1:.+]] = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<4x4xbf16>, tensor<4xbf16>) outs(%{{.+}} : tensor<4x4xbf16>) {
    // CHECK: ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
    // CHECK:  %3 = arith.addf %in, %in_0 : bf16
    // CHECK:  linalg.yield %3 : bf16
    // CHECK: } -> tensor<4x4xbf16>
    // CHECK: %[[temp2:.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%[[temp1]] : tensor<4x4xbf16>) outs(%arg2 : tensor<4x4xbf16>) {
    // CHECK: ^bb0(%in: bf16, %out: bf16):
    // CHECK:  %3 = arith.maxf %in, %cst : bf16
    // CHECK:  linalg.yield %3 : bf16
    // CHECK: } -> tensor<4x4xbf16>
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<4x4xbf16>, tensor<4xbf16>) outs(%arg2 : tensor<4x4xbf16>) {
    ^bb0(%in: bf16, %in_1: bf16, %out: bf16):
      %1 = arith.addf %in, %in_1 : bf16
      %2 = arith.maxf %1, %cst : bf16
      linalg.yield %2 : bf16
      } -> tensor<4x4xbf16>
    return %0 : tensor<4x4xbf16>
  }
}

