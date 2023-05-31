// RUN: tpp-opt -transform-dialect-interpreter -split-input-file -verify-diagnostics %s | FileCheck %s

transform.sequence failures(propagate) {
  ^bb0(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.collapse %0 [[0, 1], [2], [3, 4]] : !transform.any_op -> !transform.any_op
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

transform.sequence failures(propagate) {
  ^bb0(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.collapse %0 [[0, 1], [2]] : !transform.any_op -> !transform.any_op
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

// -----

// This must fail as we attempt to collapse dimensions of different types.
transform.sequence failures(propagate) {
  ^bb0(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.collapse %0 [[0, 1, 2]] : !transform.any_op -> !transform.any_op
}

#map0 = affine_map<(i, j, k) -> (i, j)>
#map1 = affine_map<(i, j, k) -> (i, k)>
#map2 = affine_map<(i, j, k) -> (k, j)>

func.func @matmul(%arg0: tensor<3x2xf32>, %arg1: tensor<2x3xf32>, %arg2: tensor<3x3xf32>) -> tensor<3x3xf32> {
  // expected-error@+1 {{invalid reassociation}}
  %0 = linalg.generic {indexing_maps = [#map1, #map2, #map0], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<3x2xf32>, tensor<2x3xf32>) outs(%arg2 : tensor<3x3xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %mul = arith.mulf %arg3, %arg4 : f32
    %add = arith.addf %arg5, %mul : f32
    linalg.yield %add : f32
  } -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

// -----

// This must fail as the reassociation dimensions do not match the number of loops.
transform.sequence failures(propagate) {
  ^bb0(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.collapse %0 [[0, 1]] : !transform.any_op -> !transform.any_op
}

#map0 = affine_map<(i, j, k) -> (i, j)>
#map1 = affine_map<(i, j, k) -> (i, k)>
#map2 = affine_map<(i, j, k) -> (k, j)>

func.func @matmul(%arg0: tensor<3x2xf32>, %arg1: tensor<2x3xf32>, %arg2: tensor<3x3xf32>) -> tensor<3x3xf32> {
  // expected-error@+1 {{invalid reassociation}}
  %0 = linalg.generic {indexing_maps = [#map1, #map2, #map0], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<3x2xf32>, tensor<2x3xf32>) outs(%arg2 : tensor<3x3xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %mul = arith.mulf %arg3, %arg4 : f32
    %add = arith.addf %arg5, %mul : f32
    linalg.yield %add : f32
  } -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

// -----

transform.sequence failures(propagate) {
  ^bb0(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.collapse %0 [[0, 1], [2]] : !transform.any_op -> !transform.any_op
}

#map0 = affine_map<(i, j, k) -> (i, j, k)>

// CHECK: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK: func.func @parallel(%[[arg0:.*]]: tensor<3x3x3xf32>, %[[arg1:.*]]: tensor<3x3x3xf32>)
func.func @parallel(%arg0: tensor<3x3x3xf32> , %arg1: tensor<3x3x3xf32>) -> tensor<3x3x3xf32> {
  // CHECK: %[[ta:.*]] = tensor.collapse_shape %[[arg0]] {{\[}}[0, 1], [2]] : tensor<3x3x3xf32> into tensor<9x3xf32>
  // CHECK: %[[tb:.*]] = tensor.collapse_shape %[[arg1]] {{\[}}[0, 1], [2]] : tensor<3x3x3xf32> into tensor<9x3xf32>
  // CHECK: linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP0]]], iterator_types = ["parallel", "parallel"]} ins(%[[ta]] : tensor<9x3xf32>) outs(%[[tb]] : tensor<9x3xf32>)
  %0 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0: tensor<3x3x3xf32>) outs(%arg1: tensor<3x3x3xf32>) {
  ^bb0(%arg3: f32, %arg4: f32):
    linalg.yield %arg3: f32
  } -> tensor<3x3x3xf32>
  return %0 : tensor<3x3x3xf32>
}

// -----

transform.sequence failures(propagate) {
  ^bb0(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.collapse %0 [[0, 1], [2]] : !transform.any_op -> !transform.any_op
}

#map0 = affine_map<(i, j, k) -> (i, j)>
#map1 = affine_map<(i, j, k) -> (i, k)>
#map2 = affine_map<(i, j, k) -> (k, j)>

func.func @matmul(%arg0: tensor<3x2xf32>, %arg1: tensor<2x3xf32>, %arg2: tensor<3x3xf32>) -> tensor<3x3xf32> {
  // expected-error@+1 {{fail to collapse}}
  %0 = linalg.generic {indexing_maps = [#map1, #map2, #map0], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<3x2xf32>, tensor<2x3xf32>) outs(%arg2 : tensor<3x3xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %mul = arith.mulf %arg3, %arg4 : f32
    %add = arith.addf %arg5, %mul : f32
    linalg.yield %add : f32
  } -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}
