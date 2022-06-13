// RUN: standalone-opt %s -split-input-file -enforce-tpp-preconditions | FileCheck %s
#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @gemm(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>, %arg2: tensor<3x3xf32>) -> tensor<3x3xf32> {
  %0 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], library_call = "tpp.matmul"} ins(%arg0, %arg1 : tensor<3x3xf32>, tensor<3x3xf32>) outs(%arg2 : tensor<3x3xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %1 = arith.mulf %arg3, %arg4 : f32
    %2 = arith.addf %arg5, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

// CHECK: #map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK: module {
// CHECK:  func.func @gemm(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>, %arg2: tensor<3x3xf32>) -> tensor<3x3xf32> {
// CHECK:    %cst = arith.constant 0.000000e+00 : f32
// CHECK:    %cst_0 = arith.constant 1.000000e+00 : f32
// CHECK:    %c0 = arith.constant 0 : index
// CHECK:    %c13 = arith.constant 13 : index
// CHECK:    %c3 = arith.constant 3 : index
// CHECK:    %0 = tensor.pad %arg2 low[%c0, %c0] high[%c0, %c13] {
// CHECK:    ^bb0(%arg3: index, %arg4: index):
// CHECK:      tensor.yield %cst : f32
// CHECK:    } : tensor<3x3xf32> to tensor<3x16xf32>
// CHECK:    %1 = tensor.pad %arg1 low[%c0, %c0] high[%c0, %c13] {
// CHECK:    ^bb0(%arg3: index, %arg4: index):
// CHECK:      tensor.yield %cst_0 : f32
// CHECK:    } : tensor<3x3xf32> to tensor<3x16xf32>
// CHECK:    %2 = tensor.pad %0 low[%c0, %c0] high[%c3, %c0] {
// CHECK:    ^bb0(%arg3: index, %arg4: index):
// CHECK:      tensor.yield %cst : f32
// CHECK:    } : tensor<3x16xf32> to tensor<6x16xf32>
// CHECK:    %3 = tensor.pad %arg0 low[%c0, %c0] high[%c3, %c0] {
// CHECK:    ^bb0(%arg3: index, %arg4: index):
// CHECK:      tensor.yield %cst_0 : f32
// CHECK:    } : tensor<3x3xf32> to tensor<6x3xf32>
// CHECK:    %4 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], library_call = "tpp.matmul"} ins(%3, %1 : tensor<6x3xf32>, tensor<3x16xf32>) outs(%2 : tensor<6x16xf32>) {
// CHECK:    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
// CHECK:      %6 = arith.mulf %arg3, %arg4 : f32
// CHECK:      %7 = arith.addf %arg5, %6 : f32
// CHECK:      linalg.yield %7 : f32
// CHECK:    } -> tensor<6x16xf32>
// CHECK:    %5 = tensor.extract_slice %4[0, 0] [3, 3] [1, 1] : tensor<6x16xf32> to tensor<3x3xf32>
// CHECK:    return %5 : tensor<3x3xf32>
// CHECK:  }
// CHECK:}
