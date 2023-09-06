// RUN: tpp-opt -transform-dialect-interpreter -verify-diagnostics -split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @propagation(%arg0: tensor<12x2x56x56x32xf32>) -> tensor<12x56x56x64xf32> {
  %0 = tensor.empty() : tensor<12x56x56x64xf32>
  %1 = tensor.unpack %arg0 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %0 : tensor<12x2x56x56x32xf32> -> tensor<12x56x56x64xf32>
  %c0 = arith.constant 0.0 : f32
  %2 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1 : tensor<12x56x56x64xf32>) {
  ^bb0(%out: f32):
    %3 = arith.maximumf %out, %c0 : f32
    linalg.yield %3 : f32
  } -> tensor<12x56x56x64xf32>
  return %2 : tensor<12x56x56x64xf32>
}

transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.packing_propagation %0 : !transform.any_op
}

// CHECK: func.func @propagation(
// CHECK-SAME: %[[ARG0:[0-9a-z]+]]: tensor<12x2x56x56x32xf32>) -> tensor<12x56x56x64xf32> {
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<12x56x56x64xf32>
// CHECK: %[[RELU:.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} outs(%[[ARG0]] : tensor<12x2x56x56x32xf32>)
// CHECK: %[[UNPACK:.+]] = tensor.unpack %[[RELU]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %0 : tensor<12x2x56x56x32xf32> -> tensor<12x56x56x64xf32>
// CHECK: return %[[UNPACK]] : tensor<12x56x56x64xf32>
// CHECK: }

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @propagation1(%arg0: tensor<12x2x56x56x32xf32>) -> tensor<12x56x56x64xf32> {
  %0 = tensor.empty() : tensor<12x56x56x64xf32>
  %1 = tensor.unpack %arg0 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %0 : tensor<12x2x56x56x32xf32> -> tensor<12x56x56x64xf32>
  %c0 = arith.constant 0.0 : f32
  %2 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1 : tensor<12x56x56x64xf32>) {
  ^bb0(%out: f32):
    %3 = arith.maximumf %out, %c0 : f32
    linalg.yield %3 : f32
  } -> tensor<12x56x56x64xf32>
  return %2 : tensor<12x56x56x64xf32>
}

transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = get_parent_op %0 : (!transform.any_op) -> !transform.any_op
    transform.structured.packing_propagation %1 : !transform.any_op
}

// CHECK: func.func @propagation1(
// CHECK-SAME: %[[ARG0:[0-9a-z]+]]: tensor<12x2x56x56x32xf32>) -> tensor<12x56x56x64xf32> {
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<12x56x56x64xf32>
// CHECK: %[[RELU:.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} outs(%[[ARG0]] : tensor<12x2x56x56x32xf32>)
// CHECK: %[[UNPACK:.+]] = tensor.unpack %[[RELU]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %0 : tensor<12x2x56x56x32xf32> -> tensor<12x56x56x64xf32>
// CHECK: return %[[UNPACK]] : tensor<12x56x56x64xf32>
// CHECK: }

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @main(%arg0: tensor<12x2x56x56x32xf32>) -> tensor<12x56x56x64xf32> {
  %0 = tensor.empty() : tensor<12x56x56x64xf32>
  %1 = tensor.unpack %arg0 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %0 : tensor<12x2x56x56x32xf32> -> tensor<12x56x56x64xf32>
  %c0 = arith.constant 0.0 : f32
  // expected-note @below {{non-isolated target}}
  %2 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1 : tensor<12x56x56x64xf32>) {
  ^bb0(%out: f32):
    %3 = arith.maximumf %out, %c0 : f32
    linalg.yield %3 : f32
  } -> tensor<12x56x56x64xf32>
  return %2 : tensor<12x56x56x64xf32>
}

transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{op requires isolated-from-above targets}}
    transform.structured.packing_propagation %0 : !transform.any_op
}

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @matmul(%arg0: tensor<128x512xf32>, %arg1: tensor<512x256xf32>, %arg2: tensor<128x256xf32>) -> tensor<128x256xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1: tensor<128x512xf32>, tensor<512x256xf32>) outs(%arg2: tensor<128x256xf32>) -> tensor<128x256xf32>
  %c0 = arith.constant 0.0 : f32
  %1 = linalg.generic {indexing_maps = [#map0], iterator_types = ["parallel", "parallel"]} outs(%0: tensor<128x256xf32>) {
    ^bb0(%arg3: f32):
      %2 = arith.maximumf %arg3, %c0 : f32
      linalg.yield %2 : f32
  } -> tensor<128x256xf32>
  return %1 : tensor<128x256xf32>
}

transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.pack_ext %0 blocking_factors = [32, 32, 32] : !transform.any_op -> !transform.any_op 
    %2 = get_parent_op %1 : (!transform.any_op) -> !transform.any_op
    transform.structured.packing_propagation %2 : !transform.any_op
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: func.func @matmul(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<128x512xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: tensor<512x256xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: tensor<128x256xf32>) -> tensor<128x256xf32> {
// CHECK: %[[BUFF0:.+]] = tensor.empty() : tensor<4x16x32x32xf32>
// CHECK: %[[PACK0:.+]] = tensor.pack %[[ARG0]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUFF0]] : tensor<128x512xf32> -> tensor<4x16x32x32xf32>
// CHECK: %[[BUFF1:.+]] = tensor.empty() : tensor<8x16x32x32xf32>
// CHECK: %[[PACK1:.+]] = tensor.pack %[[ARG1]] outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUFF1]] : tensor<512x256xf32> -> tensor<8x16x32x32xf32>
// CHECK: %[[BUFF2:.+]] = tensor.empty() : tensor<4x8x32x32xf32>
// CHECK: %[[PACK2:.+]] = tensor.pack %[[ARG2]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUFF2]] : tensor<128x256xf32> -> tensor<4x8x32x32xf32>
// CHECK: %[[VAL:.+]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%[[PACK0]], %[[PACK1]] : tensor<4x16x32x32xf32>, tensor<8x16x32x32xf32>) outs(%[[PACK2]] : tensor<4x8x32x32xf32>)
// CHECK: %[[VAL1:.+]] = linalg.generic {indexing_maps = [#[[MAP3]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%[[VAL]] : tensor<4x8x32x32xf32>)
// CHECK: %[[OUT:.+]] = tensor.unpack %[[VAL1]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[ARG2]] : tensor<4x8x32x32xf32> -> tensor<128x256xf32>
// CHECK: return %[[OUT]] : tensor<128x256xf32>
// CHECK: }

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @matmul(%arg0: tensor<128x512xf32>, %arg1: tensor<512x256xf32>, %arg2: tensor<128x256xf32>) -> tensor<128x256xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1: tensor<128x512xf32>, tensor<512x256xf32>) outs(%arg2: tensor<128x256xf32>) -> tensor<128x256xf32>
  %1 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%0: tensor<128x256xf32>) outs(%arg2: tensor<128x256xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      %2 = arith.addf %arg3, %arg4 : f32
      linalg.yield %2 : f32
  } -> tensor<128x256xf32>
  return %1 : tensor<128x256xf32>
}

transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.pack_ext %0 blocking_factors = [32, 32, 32] : !transform.any_op -> !transform.any_op 
    %2 = get_parent_op %1 : (!transform.any_op) -> !transform.any_op
    transform.structured.packing_propagation %2 : !transform.any_op
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: func.func @matmul(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<128x512xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: tensor<512x256xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: tensor<128x256xf32>) -> tensor<128x256xf32> {
// CHECK: %[[BUFF0:.+]] = tensor.empty() : tensor<4x16x32x32xf32>
// CHECK: %[[PACK0:.+]] = tensor.pack %[[ARG0]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUFF0]] : tensor<128x512xf32> -> tensor<4x16x32x32xf32>
// CHECK: %[[BUFF1:.+]] = tensor.empty() : tensor<8x16x32x32xf32>
// CHECK: %[[PACK1:.+]] = tensor.pack %[[ARG1]] outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUFF1]] : tensor<512x256xf32> -> tensor<8x16x32x32xf32>
// CHECK: %[[BUFF2:.+]] = tensor.empty() : tensor<4x8x32x32xf32>
// CHECK: %[[PACK2:.+]] = tensor.pack %[[ARG2]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUFF2]] : tensor<128x256xf32> -> tensor<4x8x32x32xf32>
// CHECK: %[[VAL:.+]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%[[PACK0]], %[[PACK1]] : tensor<4x16x32x32xf32>, tensor<8x16x32x32xf32>) outs(%[[PACK2]] : tensor<4x8x32x32xf32>)
// CHECK: %[[BUFF2_2:.+]] = tensor.empty() : tensor<4x8x32x32xf32>
// CHECK: %[[PACK2_2:.+]] = tensor.pack %[[ARG2]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUFF2_2]] : tensor<128x256xf32> -> tensor<4x8x32x32xf32>
// CHECK: %[[VAL1:.+]] = linalg.generic {indexing_maps = [#[[MAP3]], #[[MAP3]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[VAL]] : tensor<4x8x32x32xf32>) outs(%[[PACK2_2]] : tensor<4x8x32x32xf32>)
// CHECK: %[[OUT:.+]] = tensor.unpack %[[VAL1]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[ARG2]] : tensor<4x8x32x32xf32> -> tensor<128x256xf32>
// CHECK: return %[[OUT]] : tensor<128x256xf32>
// CHECK: }

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @conv(%arg0: tensor<1x56x56x64xf32>, %arg1: tensor<1x1x64x64xf32>, %arg2: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf ins(%arg0, %arg1: tensor<1x56x56x64xf32>, tensor<1x1x64x64xf32>) outs(%arg2: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %c0 = arith.constant 0.0 : f32
  %1 = linalg.generic {indexing_maps = [#map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%0 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32):
      %13 = arith.maximumf %in, %c0 : f32
      linalg.yield %13 : f32
  } -> tensor<1x56x56x64xf32>
  return %1 : tensor<1x56x56x64xf32>
}

transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.pack_ext %0 blocking_factors = [32, 32] : !transform.any_op -> !transform.any_op 
    %2 = get_parent_op %1 : (!transform.any_op) -> !transform.any_op
    transform.structured.packing_propagation %2 : !transform.any_op
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK: func.func @conv(
// CHECK-SAME: %[[ARG0:.+]]: tensor<1x56x56x64xf32>,
// CHECK-SAME: %[[ARG1:.+]]: tensor<1x1x64x64xf32>,
// CHECK-SAME: %[[ARG2:.+]]: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32> {
// CHECK: %[[BUFF0:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK0:.+]] = tensor.pack %[[ARG0]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF0]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CHECK: %[[BUFF1:.+]] = tensor.empty() : tensor<2x2x1x1x32x32xf32>
// CHECK: %[[PACK1:.+]] = tensor.pack %[[ARG1]] outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %[[BUFF1]] : tensor<1x1x64x64xf32> -> tensor<2x2x1x1x32x32xf32>
// CHECK: %[[BUFF2:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK2:.+]] = tensor.pack %[[ARG2]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF2]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CHECK: %[[VAL:.+]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%[[PACK0]], %[[PACK1]] : tensor<1x2x56x56x32xf32>, tensor<2x2x1x1x32x32xf32>) outs(%[[PACK2]] : tensor<1x2x56x56x32xf32>)
// CHECK: %[[VAL1:.+]] = linalg.generic {indexing_maps = [#[[MAP3]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} outs(%[[VAL]] : tensor<1x2x56x56x32xf32>)
// CHECK: %[[UNPACK:.+]] = tensor.unpack %[[VAL1]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[ARG2]] : tensor<1x2x56x56x32xf32> -> tensor<1x56x56x64xf32>
// CHECK: return %[[UNPACK]] : tensor<1x56x56x64xf32>

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @conv(%arg0: tensor<1x56x56x64xf32>, %arg1: tensor<1x1x64x64xf32>, %arg2: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf ins(%arg0, %arg1: tensor<1x56x56x64xf32>, tensor<1x1x64x64xf32>) outs(%arg2: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %1 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%0 : tensor<1x56x56x64xf32>) outs(%arg2: tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %13 = arith.addf %in, %out : f32
      linalg.yield %13 : f32
  } -> tensor<1x56x56x64xf32>
  return %1 : tensor<1x56x56x64xf32>
}

transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.pack_ext %0 blocking_factors = [32, 32] : !transform.any_op -> !transform.any_op 
    %2 = get_parent_op %1 : (!transform.any_op) -> !transform.any_op
    transform.structured.packing_propagation %2 : !transform.any_op
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK: func.func @conv(
// CHECK-SAME: %[[ARG0:.+]]: tensor<1x56x56x64xf32>,
// CHECK-SAME: %[[ARG1:.+]]: tensor<1x1x64x64xf32>,
// CHECK-SAME: %[[ARG2:.+]]: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32> {
// CHECK: %[[BUFF0:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK0:.+]] = tensor.pack %[[ARG0]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF0]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CHECK: %[[BUFF1:.+]] = tensor.empty() : tensor<2x2x1x1x32x32xf32>
// CHECK: %[[PACK1:.+]] = tensor.pack %[[ARG1]] outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %[[BUFF1]] : tensor<1x1x64x64xf32> -> tensor<2x2x1x1x32x32xf32>
// CHECK: %[[BUFF2:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK2:.+]] = tensor.pack %[[ARG2]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF2]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CHECK: %[[VAL:.+]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%[[PACK0]], %[[PACK1]] : tensor<1x2x56x56x32xf32>, tensor<2x2x1x1x32x32xf32>) outs(%[[PACK2]] : tensor<1x2x56x56x32xf32>)
// CHECK: %[[BUFF2_2:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK2_2:.+]] = tensor.pack %[[ARG2]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF2_2]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CHECK: %[[VAL1:.+]] = linalg.generic {indexing_maps = [#[[MAP3]], #[[MAP3]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%[[VAL]] : tensor<1x2x56x56x32xf32>) outs(%[[PACK2_2]] : tensor<1x2x56x56x32xf32>)
// CHECK: %[[UNPACK:.+]] = tensor.unpack %[[VAL1]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[ARG2]] : tensor<1x2x56x56x32xf32> -> tensor<1x56x56x64xf32>
// CHECK: return %[[UNPACK]] : tensor<1x56x56x64xf32>

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3)>

func.func @conv(%arg0: tensor<1x56x56x64xf32>, %arg1: tensor<1x1x64x64xf32>, %arg2: tensor<1x56x56x64xf32>, %arg3: tensor<64xf32>) -> tensor<1x56x56x64xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf ins(%arg0, %arg1: tensor<1x56x56x64xf32>, tensor<1x1x64x64xf32>) outs(%arg2: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %1 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%0, %arg3 : tensor<1x56x56x64xf32>, tensor<64xf32>) outs(%arg2: tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %168 = arith.addf %in, %in_1 : f32
      linalg.yield %168 : f32
  } -> tensor<1x56x56x64xf32>
  return %1 : tensor<1x56x56x64xf32>
}

transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.pack_ext %0 blocking_factors = [32, 32] : !transform.any_op -> !transform.any_op 
    %2 = get_parent_op %1 : (!transform.any_op) -> !transform.any_op
    transform.structured.packing_propagation %2 : !transform.any_op
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d4)>
// CHECK: func.func @conv(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]*]]: tensor<1x56x56x64xf32>,
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9]*]]: tensor<1x1x64x64xf32>,
// CHECK-SAME: %[[ARG2:[a-zA-Z0-9]*]]: tensor<1x56x56x64xf32>,
// CHECK-SAME: %[[ARG3:[a-zA-Z0-9]*]]: tensor<64xf32>) -> tensor<1x56x56x64xf32>
// CHECK: %[[BUFF0:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK0:.+]] = tensor.pack %[[ARG0]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF0]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CHECK: %[[BUFF1:.+]] = tensor.empty() : tensor<2x2x1x1x32x32xf32>
// CHECK: %[[PACK1:.+]] = tensor.pack %[[ARG1]] outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %[[BUFF1]] : tensor<1x1x64x64xf32> -> tensor<2x2x1x1x32x32xf32>
// CHECK: %[[BUFF2:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK2:.+]] = tensor.pack %[[ARG2]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF2]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CHECK: %[[VAL:.+]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%[[PACK0]], %[[PACK1]] : tensor<1x2x56x56x32xf32>, tensor<2x2x1x1x32x32xf32>) outs(%[[PACK2]] : tensor<1x2x56x56x32xf32>)
// CHECK: %[[BUFF3:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK3:.+]] = tensor.pack %[[ARG2]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF3]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CHECK: %[[EXPAND:.+]] = tensor.expand_shape %[[ARG3]] {{\[}}[0, 1]] : tensor<64xf32> into tensor<2x32xf32>
// CHECK: %[[VAL1:.+]] = linalg.generic {indexing_maps = [#[[MAP3]], #[[MAP4]], #[[MAP3]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%[[VAL]], %[[EXPAND]] : tensor<1x2x56x56x32xf32>, tensor<2x32xf32>) outs(%[[PACK3]] : tensor<1x2x56x56x32xf32>)
// CHECK: %[[UNPACK:.+]] = tensor.unpack %[[VAL1]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[ARG2]] : tensor<1x2x56x56x32xf32> -> tensor<1x56x56x64xf32>

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>

func.func @conv(%arg0: tensor<1x56x56x64xf32>, %arg1: tensor<1x1x64x64xf32>, %arg2: tensor<1x56x56x64xf32>, %arg3: tensor<56x64xf32>) -> tensor<1x56x56x64xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf ins(%arg0, %arg1: tensor<1x56x56x64xf32>, tensor<1x1x64x64xf32>) outs(%arg2: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %1 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%0, %arg3 : tensor<1x56x56x64xf32>, tensor<56x64xf32>) outs(%arg2: tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %168 = arith.addf %in, %in_1 : f32
      linalg.yield %168 : f32
  } -> tensor<1x56x56x64xf32>
  return %1 : tensor<1x56x56x64xf32>
}

transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.pack_ext %0 blocking_factors = [32, 32] : !transform.any_op -> !transform.any_op 
    %2 = get_parent_op %1 : (!transform.any_op) -> !transform.any_op
    transform.structured.packing_propagation %2 : !transform.any_op
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>
// CHECK: func.func @conv(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]*]]: tensor<1x56x56x64xf32>,
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9]*]]: tensor<1x1x64x64xf32>,
// CHECK-SAME: %[[ARG2:[a-zA-Z0-9]*]]: tensor<1x56x56x64xf32>,
// CHECK-SAME: %[[ARG3:[a-zA-Z0-9]*]]: tensor<56x64xf32>) -> tensor<1x56x56x64xf32>
// CHECK: %[[BUFF0:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK0:.+]] = tensor.pack %[[ARG0]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF0]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CHECK: %[[BUFF1:.+]] = tensor.empty() : tensor<2x2x1x1x32x32xf32>
// CHECK: %[[PACK1:.+]] = tensor.pack %[[ARG1]] outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %[[BUFF1]] : tensor<1x1x64x64xf32> -> tensor<2x2x1x1x32x32xf32>
// CHECK: %[[BUFF2:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK2:.+]] = tensor.pack %[[ARG2]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF2]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CHECK: %[[VAL:.+]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%[[PACK0]], %[[PACK1]] : tensor<1x2x56x56x32xf32>, tensor<2x2x1x1x32x32xf32>) outs(%[[PACK2]] : tensor<1x2x56x56x32xf32>)
// CHECK: %[[BUFF4:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK4:.+]] = tensor.pack %[[ARG2]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF4]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CHECK: %[[BUFF3:.+]] = tensor.empty() : tensor<2x56x32xf32>
// CHECK: %[[PACK3:.+]] = tensor.pack %[[ARG3]] outer_dims_perm = [1, 0] inner_dims_pos = [1] inner_tiles = [32] into %[[BUFF3]] : tensor<56x64xf32> -> tensor<2x56x32xf32>
// CHECK: %[[VAL1:.+]] = linalg.generic {indexing_maps = [#[[MAP3]], #[[MAP4]], #[[MAP3]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%[[VAL]], %[[PACK3]] : tensor<1x2x56x56x32xf32>, tensor<2x56x32xf32>) outs(%[[PACK4]] : tensor<1x2x56x56x32xf32>)
// CHECK: %[[UNPACK:.+]] = tensor.unpack %[[VAL1]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[ARG2]] : tensor<1x2x56x56x32xf32> -> tensor<1x56x56x64xf32>

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @conv(%arg0: tensor<1x56x56x64xf32>, %arg1: tensor<1x1x64x64xf32>, %arg2: tensor<1x56x56x64xf32>) -> tensor<1x58x58x64xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf ins(%arg0, %arg1: tensor<1x56x56x64xf32>, tensor<1x1x64x64xf32>) outs(%arg2: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %c0 = arith.constant 0.0 : f32
  %1 = linalg.generic {indexing_maps = [#map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%0 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32):
      %13 = arith.maximumf %in, %c0 : f32
      linalg.yield %13 : f32
  } -> tensor<1x56x56x64xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %2 = tensor.pad %1 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
      tensor.yield %cst : f32
  } : tensor<1x56x56x64xf32> to tensor<1x58x58x64xf32>
  return %2 : tensor<1x58x58x64xf32>
}

transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.pack_ext %0 blocking_factors = [32, 32] : !transform.any_op -> !transform.any_op 
    %2 = get_parent_op %1 : (!transform.any_op) -> !transform.any_op
    transform.structured.packing_propagation %2 : !transform.any_op
}

// CONV-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
// CONV-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
// CONV-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
// CONV-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CONV: func.func @conv(
// CONV-SAME: %[[ARG0:.+]]: tensor<1x56x56x64xf32>,
// CONV-SAME: %[[ARG1:.+]]: tensor<1x1x64x64xf32>,
// CONV-SAME: %[[ARG2:.+]]: tensor<1x56x56x64xf32>) -> tensor<1x58x58x64xf32> {
// CONV: %[[BUFF0:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CONV: %[[PACK0:.+]] = tensor.pack %[[ARG0]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF0]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CONV: %[[BUFF1:.+]] = tensor.empty() : tensor<2x2x1x1x32x32xf32>
// CONV: %[[PACK1:.+]] = tensor.pack %[[ARG1]] outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %[[BUFF1]] : tensor<1x1x64x64xf32> -> tensor<2x2x1x1x32x32xf32>
// CONV: %[[BUFF2:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CONV: %[[PACK2:.+]] = tensor.pack %[[ARG2]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF2]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CONV: %[[VAL:.+]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%[[PACK0]], %[[PACK1]] : tensor<1x2x56x56x32xf32>, tensor<2x2x1x1x32x32xf32>) outs(%[[PACK2]] : tensor<1x2x56x56x32xf32>)
// CONV: %[[VAL1:.+]] = linalg.generic {indexing_maps = [#[[MAP3]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} outs(%[[VAL]] : tensor<1x2x56x56x32xf32>)
// CONV: %[[PADDED:.+]] = tensor.pad %[[VAL1]] low[0, 0, 1, 1, 0] high[0, 0, 1, 1, 0]
// CONV: %[[OUT:.+]] = tensor.empty() : tensor<1x58x58x64xf32>
// CONV: %[[UNPACK:.+]] = tensor.unpack %[[PADDED]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[OUT]] : tensor<1x2x58x58x32xf32> -> tensor<1x58x58x64xf32>
// CONV: return %[[UNPACK]] : tensor<1x58x58x64xf32>
