// RUN: tpp-opt -transform-dialect-interpreter -verify-diagnostics -split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @propagation(%arg0: tensor<12x2x56x56x32xf32>) -> tensor<12x56x56x64xf32> {
  %0 = tensor.empty() : tensor<12x56x56x64xf32>
  %1 = linalgx.unpack %arg0 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %0 : (tensor<12x2x56x56x32xf32> tensor<12x56x56x64xf32>) -> tensor<12x56x56x64xf32>
  %2 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1 : tensor<12x56x56x64xf32>) {
  ^bb0(%out: f32):
    %3 = mathx.relu %out : f32
    linalg.yield %3 : f32
  } -> tensor<12x56x56x64xf32>
  return %2 : tensor<12x56x56x64xf32>
}
  
transform.sequence failures(propagate) {
  ^bb0(%arg0: !pdl.operation):
    %0 = transform.structured.match ops{["func.func"]} in %arg0
    transform.structured.packing_propagation %0
}

// CHECK: func.func @propagation(
// CHECK-SAME: %[[ARG0:[0-9a-z]+]]: tensor<12x2x56x56x32xf32>) -> tensor<12x56x56x64xf32> {
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<12x56x56x64xf32>
// CHECK: %[[RELU:.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} outs(%[[ARG0]] : tensor<12x2x56x56x32xf32>)
// CHECK: %[[UNPACK:.+]] = linalgx.unpack %[[RELU]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %0 : (tensor<12x2x56x56x32xf32> tensor<12x56x56x64xf32>) -> tensor<12x56x56x64xf32>
// CHECK: return %[[UNPACK]] : tensor<12x56x56x64xf32>
// CHECK: }

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @propagation1(%arg0: tensor<12x2x56x56x32xf32>) -> tensor<12x56x56x64xf32> {
  %0 = tensor.empty() : tensor<12x56x56x64xf32>
  %1 = linalgx.unpack %arg0 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %0 : (tensor<12x2x56x56x32xf32> tensor<12x56x56x64xf32>) -> tensor<12x56x56x64xf32>
  %2 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1 : tensor<12x56x56x64xf32>) { 
  ^bb0(%out: f32):
    %3 = mathx.relu %out : f32
    linalg.yield %3 : f32
  } -> tensor<12x56x56x64xf32>
  return %2 : tensor<12x56x56x64xf32>
} 

transform.sequence failures(propagate) {
  ^bb0(%arg0: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg0
    %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
    transform.structured.packing_propagation %1
}

// CHECK: func.func @propagation1(
// CHECK-SAME: %[[ARG0:[0-9a-z]+]]: tensor<12x2x56x56x32xf32>) -> tensor<12x56x56x64xf32> {
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<12x56x56x64xf32>
// CHECK: %[[RELU:.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} outs(%[[ARG0]] : tensor<12x2x56x56x32xf32>)
// CHECK: %[[UNPACK:.+]] = linalgx.unpack %[[RELU]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %0 : (tensor<12x2x56x56x32xf32> tensor<12x56x56x64xf32>) -> tensor<12x56x56x64xf32>
// CHECK: return %[[UNPACK]] : tensor<12x56x56x64xf32>
// CHECK: }

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @main(%arg0: tensor<12x2x56x56x32xf32>) -> tensor<12x56x56x64xf32> {
  %0 = tensor.empty() : tensor<12x56x56x64xf32>
  %1 = linalgx.unpack %arg0 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %0 : (tensor<12x2x56x56x32xf32> tensor<12x56x56x64xf32>) -> tensor<12x56x56x64xf32>
  // expected-note @below {{non-isolated target}}
  %2 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1 : tensor<12x56x56x64xf32>) { 
  ^bb0(%out: f32):
    %3 = mathx.relu %out : f32
    linalg.yield %3 : f32
  } -> tensor<12x56x56x64xf32>
  return %2 : tensor<12x56x56x64xf32>
} 

transform.sequence failures(propagate) {
  ^bb0(%arg0: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg0
    // expected-error @below {{op requires isolated-from-above targets}}
    transform.structured.packing_propagation %0
}
