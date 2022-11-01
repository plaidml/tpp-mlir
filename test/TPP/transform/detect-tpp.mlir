// RUN: tpp-opt -transform-dialect-interpreter -verify-diagnostics -split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @relu(%arg0: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %0 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"], library_call = "tpp.relu"} outs(%arg0: tensor<128x128xf32>) {
    ^bb0(%out: f32):
      %1 = mathx.relu %out : f32
      linalg.yield %1 : f32
  } -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["func.func"]} in %arg1
    transform.structured.map_linalg_to_tpp %0
}

// CHECK: func.func @relu(
// CHECK-SAME: %[[ARG0:[0-9a-z]+]]: tensor<128x128xf32>) -> tensor<128x128xf32> {
// CHECK: {{.*}} = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"], library_call = "tpp.relu"} outs(%[[ARG0]] : tensor<128x128xf32>)

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @relu(%arg0: tensor<128x128xf32>) -> tensor<128x128xf32> {
  // expected-note @below {{non-isolated target}}
  %0 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"], library_call = "tpp.relu"} outs(%arg0: tensor<128x128xf32>) {
    ^bb0(%out: f32):
      %1 = mathx.relu %out : f32
      linalg.yield %1 : f32
  } -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
    // expected-error @below {{op requires isolated-from-above targets}}
    transform.structured.map_linalg_to_tpp %0
}
