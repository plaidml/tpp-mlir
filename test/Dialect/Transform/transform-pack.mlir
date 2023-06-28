// RUN: tpp-opt -transform-dialect-interpreter -verify-diagnostics -split-input-file %s | FileCheck %s

transform.sequence failures(propagate) {
^bb1(%arg0: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  // CHECK: transform.structured.pack_ext
  %1 = transform.structured.pack_ext %0 blocking_factors = [2, 2, 2] : !transform.any_op -> !transform.any_op 
}

// -----

#mapO = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#mapI = affine_map<(d0, d1, d2) -> (d2, d1, d0)>

func.func @parallel(%arg0: tensor<5x5x5xf32>, %arg1: tensor<5x5x5xf32>) -> tensor<5x5x5xf32> {
  // expected-note @below {{when applied to this op}}
  %0 = linalg.generic {indexing_maps = [#mapI, #mapO], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0: tensor<5x5x5xf32>) outs(%arg1: tensor<5x5x5xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      linalg.yield %arg2 : f32
  } -> tensor<5x5x5xf32>
  return %0 : tensor<5x5x5xf32>
}

transform.sequence failures(propagate) {
  ^bb0(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{Could not pack op}}
    %1 = transform.structured.pack_ext %0 blocking_factors = [2, 2, 2] : !transform.any_op -> !transform.any_op 
}

// -----

transform.sequence failures(propagate) {
  ^bb0(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{Could not pack op:}}
    %1 = transform.structured.pack_ext %0 blocking_factors = [200, 200, 200] : !transform.any_op -> !transform.any_op 
}

func.func @block_linalg_matmul(
  %arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>)
    -> tensor<?x?xf32> {
  // expected-note @below {{when applied to this op}}
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%arg2: tensor<?x?xf32>)
    -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

transform.sequence failures(propagate) {
  ^bb0(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.pack_ext %0 blocking_factors = [32, 32] : !transform.any_op -> !transform.any_op 
}

func.func @conv_2d_nchw_fchw(%i: tensor<14x512x28x28xf32>, %f: tensor<1024x512x1x1xf32>,
                %o: tensor<14x1024x28x28xf32>) -> tensor<14x1024x28x28xf32> {
  %0 = linalg.conv_2d_nchw_fchw ins(%i, %f: tensor<14x512x28x28xf32>, tensor<1024x512x1x1xf32>) outs(%o: tensor<14x1024x28x28xf32>) -> tensor<14x1024x28x28xf32>
  return %0: tensor<14x1024x28x28xf32>
}


// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>

// CHECK: func.func @conv_2d_nchw_fchw(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<14x512x28x28xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: tensor<1024x512x1x1xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: tensor<14x1024x28x28xf32>) -> tensor<14x1024x28x28xf32> {
// CHECK: %[[BUF0:.+]] = tensor.empty() : tensor<14x16x28x28x32xf32>
// CHECK: %[[PACK0:.+]] = tensor.pack %[[ARG0]] inner_dims_pos = [1] inner_tiles = [32] into %[[BUF0]] : tensor<14x512x28x28xf32> -> tensor<14x16x28x28x32xf32>
// CHECK: %[[BUF1:.+]] = tensor.empty() : tensor<32x16x1x1x32x32xf32>
// CHECK: %[[PACK1:.+]] = tensor.pack %[[ARG1]] inner_dims_pos = [1, 0] inner_tiles = [32, 32] into %[[BUF1]] : tensor<1024x512x1x1xf32> -> tensor<32x16x1x1x32x32xf32>
// CHECK: %[[BUF2:.+]] = tensor.empty() : tensor<14x32x28x28x32xf32>
// CHECK: %[[PACK2:.+]] = tensor.pack %[[ARG2]] inner_dims_pos = [1] inner_tiles = [32] into %[[BUF2]] : tensor<14x1024x28x28xf32> -> tensor<14x32x28x28x32xf32>
// CHECK: %[[VAL:.+]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%[[PACK0]], %[[PACK1]] : tensor<14x16x28x28x32xf32>, tensor<32x16x1x1x32x32xf32>) outs(%[[PACK2]] : tensor<14x32x28x28x32xf32>)
// CHECK: %[[OUT:.+]] = tensor.unpack %[[VAL]] inner_dims_pos = [1] inner_tiles = [32] into %[[ARG2]] : tensor<14x32x28x28x32xf32> -> tensor<14x1024x28x28xf32>
// CHECK: return %[[OUT]] : tensor<14x1024x28x28xf32>
// CHECK: }

// -----

transform.sequence failures(propagate) {
  ^bb0(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.pack_ext %0 blocking_factors = [32, 32] : !transform.any_op -> !transform.any_op 
}

func.func @conv_2d_nhwc_hwcf(%arg0: tensor<1x113x113x64xf32>, %arg1: tensor<3x3x64x256xf32>, %arg2: tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32> {
  %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>,
                                 strides = dense<1> : tensor<2xi64>}
    ins(%arg0, %arg1 : tensor<1x113x113x64xf32>, tensor<3x3x64x256xf32>)
    outs(%arg2: tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
  return %1 : tensor<1x56x56x256xf32>
}

// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
// CHECK: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
// CHECK: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>

// CHECK: func.func @conv_2d_nhwc_hwcf(
// CHECK-SAME: %[[ARG0:.+]]: tensor<1x113x113x64xf32>,
// CHECK-SAME: %[[ARG1:.+]]: tensor<3x3x64x256xf32>,
// CHECK-SAME: %[[ARG2:.+]]: tensor<1x56x56x256xf32>
// CHECK: %[[BUF0:.+]] = tensor.empty() : tensor<1x2x113x113x32xf32>
// CHECK: %[[PACK0:.+]] = tensor.pack %[[ARG0]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUF0]] : tensor<1x113x113x64xf32> -> tensor<1x2x113x113x32xf32>
// CHECK: %[[BUF1:.+]] = tensor.empty() : tensor<8x2x3x3x32x32xf32>
// CHECK: %[[PACK1:.+]] = tensor.pack %[[ARG1]] outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %[[BUF1]] : tensor<3x3x64x256xf32> -> tensor<8x2x3x3x32x32xf32>
// CHECK: %[[BUF2:.+]] = tensor.empty() : tensor<1x8x56x56x32xf32>
// CHECK: %[[PACK2:.+]] = tensor.pack %[[ARG2]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUF2]] : tensor<1x56x56x256xf32> -> tensor<1x8x56x56x32xf32>
// CHECK: %[[GEN:.+]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%[[PACK0]], %[[PACK1]] : tensor<1x2x113x113x32xf32>, tensor<8x2x3x3x32x32xf32>) outs(%[[PACK2]] : tensor<1x8x56x56x32xf32>)
// CHECK: %[[RES:.+]] = tensor.unpack %[[GEN]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[ARG2]] : tensor<1x8x56x56x32xf32> -> tensor<1x56x56x256xf32>
// CHECK: return %[[RES]] : tensor<1x56x56x256xf32>

// -----

transform.sequence failures(propagate) {
  ^bb0(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.pack_ext %0 blocking_factors = [32, 32] : !transform.any_op -> !transform.any_op 
}

// test we properly pass stride information.
func.func @conv_2d_nhwc_hwcf(%arg0: tensor<1x113x113x64xf32>, %arg1: tensor<3x3x64x256xf32>, %arg2: tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32> {
  %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>,
                                 strides = dense<2> : tensor<2xi64>}
    ins(%arg0, %arg1 : tensor<1x113x113x64xf32>, tensor<3x3x64x256xf32>)
    outs(%arg2: tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
  return %1 : tensor<1x56x56x256xf32>
}

// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 * 2 + d6, d3 * 2 + d7, d8)>
// CHECK: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
// CHECK: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>

// CHECK: func.func @conv_2d_nhwc_hwcf(
// CHECK-SAME: %[[ARG0:.+]]: tensor<1x113x113x64xf32>,
// CHECK-SAME: %[[ARG1:.+]]: tensor<3x3x64x256xf32>,
// CHECK-SAME: %[[ARG2:.+]]: tensor<1x56x56x256xf32>
// CHECK: %[[BUF0:.+]] = tensor.empty() : tensor<1x2x113x113x32xf32>
// CHECK: %[[PACK0:.+]] = tensor.pack %[[ARG0]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUF0]] : tensor<1x113x113x64xf32> -> tensor<1x2x113x113x32xf32>
// CHECK: %[[BUF1:.+]] = tensor.empty() : tensor<8x2x3x3x32x32xf32>
// CHECK: %[[PACK1:.+]] = tensor.pack %[[ARG1]] outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %[[BUF1]] : tensor<3x3x64x256xf32> -> tensor<8x2x3x3x32x32xf32>
// CHECK: %[[BUF2:.+]] = tensor.empty() : tensor<1x8x56x56x32xf32>
// CHECK: %[[PACK2:.+]] = tensor.pack %[[ARG2]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUF2]] : tensor<1x56x56x256xf32> -> tensor<1x8x56x56x32xf32>
// CHECK: %[[GEN:.+]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%[[PACK0]], %[[PACK1]] : tensor<1x2x113x113x32xf32>, tensor<8x2x3x3x32x32xf32>) outs(%[[PACK2]] : tensor<1x8x56x56x32xf32>)
// CHECK: %[[RES:.+]] = tensor.unpack %[[GEN]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[ARG2]] : tensor<1x8x56x56x32xf32> -> tensor<1x56x56x256xf32>
// CHECK: return %[[RES]] : tensor<1x56x56x256xf32>

// -----

transform.sequence failures(propagate) {
  ^bb0(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.pack_ext %0 blocking_factors = [32, 32, 32] : !transform.any_op -> !transform.any_op 
}

func.func @matmul(%arg0: tensor<128x512xf32>, %arg1: tensor<512x256xf32>, %arg2: tensor<128x256xf32>) -> tensor<128x256xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1: tensor<128x512xf32>, tensor<512x256xf32>) outs(%arg2: tensor<128x256xf32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
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
// CHECK: %[[OUT:.+]] = tensor.unpack %[[VAL]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[ARG2]] : tensor<4x8x32x32xf32> -> tensor<128x256xf32>
// CHECK: return %[[OUT]] : tensor<128x256xf32>
// CHECK: }
