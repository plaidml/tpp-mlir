// RUN: tpp-opt --linalg-convert-compare-select-to-maximumf-pass %s --split-input-file | FileCheck %s 

func.func @forward() -> tensor<256x1024xf32>{
%cst_5 = arith.constant 0.000000e+00 : f32
%5 = tensor.empty() : tensor<256x1024xf32>
%2 = tensor.empty() : tensor<256x1024xf32>
%6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<256x1024xf32>) outs(%2 : tensor<256x1024xf32>) {
  ^bb0(%in: f32, %out: f32):
    %15 = arith.cmpf ugt, %in, %cst_5 : f32
    %16 = arith.select %15, %in, %cst_5 : f32
    linalg.yield %16 : f32
  } -> tensor<256x1024xf32>

return %6: tensor<256x1024xf32>
}

// CHECK: #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: module {
// CHECK: func.func @forward()
// CHECK: -> tensor<256x1024xf32> {
// CHECK-DAG: %[[cst:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[temp0:.*]] = tensor.empty() : tensor<256x1024xf32>
// CHECK: %[[temp1:.*]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%[[temp0]] : tensor<256x1024xf32>) {
// CHECK:    ^bb0(%[[out:.*]]: f32):
// CHECK:      %[[temp2:.*]] = arith.maximumf %[[out]], %[[cst]] : f32
// CHECK:      linalg.yield %[[temp2]] : f32
// CHECK:    } -> tensor<256x1024xf32>
// CHECK:    return %[[temp1]] : tensor<256x1024xf32>

