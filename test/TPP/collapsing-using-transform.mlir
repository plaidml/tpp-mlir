// RUN: tpp-opt -transform-dialect-interpreter -split-input-file -verify-diagnostics %s | FileCheck %s

// This must fail as we attempt to collapse dimensions of different types.
transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
    %1 = transform.structured.collapsing %0 [[0, 1, 2]]
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
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
    %1 = transform.structured.collapsing %0 [[0, 1]]
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
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
    %1 = transform.structured.collapsing %0 [[0, 1], [2]]
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
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
    %1 = transform.structured.collapsing %0 [[0, 1], [2]]
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
