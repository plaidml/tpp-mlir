// RUN: standalone-opt -transform-dialect-interpreter -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d2, d5)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d4, d5, d3)>
// CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>

#map0 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 32 + d4, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 32 + d5, d1 * 32 + d4, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d5, d2, d3, d6)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d5, d6, d4)>
#map6 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4)>
  
transform.with_pdl_patterns {
  ^bb0(%arg0: !pdl.operation):
    sequence %arg0 failures(propagate) {
      ^bb0(%arg1: !pdl.operation):
        %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
        %1 = transform.structured.collapsing %0 [[0], [1], [2, 3], [4], [5], [6]]
    }
}

// CHECK-LABEL: func.func @conv(
func.func @conv(%arg0: tensor<14x512x28x28xf32>, %arg1: tensor<1024x512x1x1xf32>, %arg2: tensor<14x1024x28x28xf32>) -> tensor<14x1024x28x28xf32> {
  %0 = tensor.empty() : tensor<14x16x28x28x32xf32>
  %1 = linalgx.relayout ins(%arg0 : tensor<14x512x28x28xf32>, #map0) outs(%0 : tensor<14x16x28x28x32xf32>, #map1) -> tensor<14x16x28x28x32xf32>
  %2 = tensor.empty() : tensor<32x16x1x1x32x32xf32>
  %3 = linalgx.relayout ins(%arg1 : tensor<1024x512x1x1xf32>, #map2) outs(%2 : tensor<32x16x1x1x32x32xf32>, #map3) -> tensor<32x16x1x1x32x32xf32>
  %4 = tensor.empty() : tensor<14x32x28x28x32xf32>
  %5 = linalgx.relayout ins(%arg2 : tensor<14x1024x28x28xf32>, #map0) outs(%4 : tensor<14x32x28x28x32xf32>, #map1) -> tensor<14x32x28x28x32xf32>
  %6 = tensor.collapse_shape %3 [[0], [1, 2, 3], [4], [5]] : tensor<32x16x1x1x32x32xf32> into tensor<32x16x32x32xf32>
  // CHECK: %{{.*}} = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins({{.*}}, %{{.*}} : tensor<14x16x784x32xf32>, tensor<32x16x32x32xf32>) outs(%{{.*}} : tensor<14x32x784x32xf32>) 
  %7 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%1, %6 : tensor<14x16x28x28x32xf32>, tensor<32x16x32x32xf32>) outs(%5 : tensor<14x32x28x28x32xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %9 = arith.mulf %arg3, %arg4 : f32
    %10 = arith.addf %arg5, %9 : f32
    linalg.yield %10 : f32
  } -> tensor<14x32x28x28x32xf32>
  %8 = linalgx.relayout ins(%7 : tensor<14x32x28x28x32xf32>, #map1) outs(%arg2 : tensor<14x1024x28x28xf32>, #map0) -> tensor<14x1024x28x28xf32>
  return %8 : tensor<14x1024x28x28xf32>
}

// -----

// This must fail as we attempt to collapse dimensions of different types.
transform.with_pdl_patterns {
  ^bb0(%arg0: !pdl.operation):
    sequence %arg0 failures(propagate) {
      ^bb0(%arg1: !pdl.operation):
        %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
        %1 = transform.structured.collapsing %0 [[0, 1, 2]]
    }
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
transform.with_pdl_patterns {
  ^bb0(%arg0: !pdl.operation):
    sequence %arg0 failures(propagate) {
      ^bb0(%arg1: !pdl.operation):
        %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
        %1 = transform.structured.collapsing %0 [[0, 1]]
    }
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

transform.with_pdl_patterns {
  ^bb0(%arg0: !pdl.operation):
    sequence %arg0 failures(propagate) {
      ^bb0(%arg1: !pdl.operation):
        %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
        %1 = transform.structured.collapsing %0 [[0, 1], [2]]
    }
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

transform.with_pdl_patterns {
  ^bb0(%arg0: !pdl.operation):
    sequence %arg0 failures(propagate) {
      ^bb0(%arg1: !pdl.operation):
        %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
        %1 = transform.structured.collapsing %0 [[0, 1], [2]]
    }
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
