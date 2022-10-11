// RUN: standalone-opt -transform-dialect-interpreter -split-input-file %s | FileCheck %s

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 failures(propagate) {
    ^bb0(%arg1: !pdl.operation):
      %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
      %1 = transform.structured.collapsing %0 [[0, 1], [2], [3, 4]]
  }
}

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

// CHECK-LABEL: func @parallel(
// CHECK-SAME: %[[TA:[0-9a-z]+]]: tensor<5x5x4x3x3xf32>
// CHECK-SAME: %[[TB:[0-9a-z]+]]: tensor<5x5x4x3x3xf32>
// CHECK-SAME: -> tensor<5x5x4x3x3xf32> {
func.func @parallel(%arg0: tensor<5x5x4x3x3xf32>, %arg1: tensor<5x5x4x3x3xf32>) -> tensor<5x5x4x3x3xf32> {
  // CHECK: %[[CA:.*]] = tensor.collapse_shape %[[TA]] {{\[}}[0, 1], [2], [3, 4]] : tensor<5x5x4x3x3xf32> into tensor<25x4x9xf32> 
  // CHECK: %[[CB:.*]] = tensor.collapse_shape %[[TB]] {{\[}}[0, 1], [2], [3, 4]] : tensor<5x5x4x3x3xf32> into tensor<25x4x9xf32>
  // CHECK: %[[res:.*]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[CA]] : tensor<25x4x9xf32>) outs(%[[CB]] : tensor<25x4x9xf32>)
  // CHECK: %[[CC:.*]] = tensor.expand_shape %[[res]] {{\[}}[0, 1], [2], [3, 4]] : tensor<25x4x9xf32> into tensor<5x5x4x3x3xf32>
  // CHECK: return %[[CC]] : tensor<5x5x4x3x3xf32>
  %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<5x5x4x3x3xf32>) outs(%arg1 : tensor<5x5x4x3x3xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
  } -> tensor<5x5x4x3x3xf32>
  return %0 : tensor<5x5x4x3x3xf32>
}

// -----

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 failures(propagate) {
    ^bb0(%arg1: !pdl.operation):
      %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
      %1 = transform.structured.collapsing %0 [[0, 1], [2]]
  }
}

// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0, d1)>
#mapO = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#mapI = affine_map<(d0, d1, d2) -> (d2, d1, d0)>

// CHECK-LABEL: func @parallel(
// CHECK-SAME: %[[TA:[0-9a-z]+]]: tensor<5x5x5xf32>
// CHECK-SAME: %[[TB:[0-9a-z]+]]: tensor<5x5x5xf32>
// CHECK-SAME: -> tensor<5x5x5xf32> {
func.func @parallel(%arg0: tensor<5x5x5xf32>, %arg1: tensor<5x5x5xf32>) -> tensor<5x5x5xf32> {
  // CHECK: %[[CA:.*]] = tensor.collapse_shape %[[TA]] {{\[}}[0], [1, 2]] : tensor<5x5x5xf32> into tensor<5x25xf32>
  // CHECK: %[[CB:.*]] = tensor.collapse_shape %[[TB]] {{\[}}[0, 1], [2]] : tensor<5x5x5xf32> into tensor<25x5xf32>
  // CHECK: %[[res:.*]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]]], iterator_types = ["parallel", "parallel"]} ins(%[[CA]] : tensor<5x25xf32>) outs(%[[CB]] : tensor<25x5xf32>)
  // CHECK: %[[CC:.*]] = tensor.expand_shape %[[res]] {{\[}}[0, 1], [2]] : tensor<25x5xf32> into tensor<5x5x5xf32>
  // CHECK: return %[[CC]] : tensor<5x5x5xf32>
  %0 = linalg.generic {indexing_maps = [#mapI, #mapO], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0: tensor<5x5x5xf32>) outs(%arg1: tensor<5x5x5xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
  } -> tensor<5x5x5xf32>
  return %0 : tensor<5x5x5xf32>
}
