// RUN: tpp-opt %s -pack-conv2DNhwcHwcf="block-factors=32,32" | FileCheck %s

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
// CHECK: %[[PACK0:.+]] = linalgx.pack %[[ARG0]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUF0]] : (tensor<1x113x113x64xf32> tensor<1x2x113x113x32xf32>) -> tensor<1x2x113x113x32xf32>
// CHECK: %[[BUF1:.+]] = tensor.empty() : tensor<8x2x3x3x32x32xf32>
// CHECK: %[[PACK1:.+]] = linalgx.pack %[[ARG1]] outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %[[BUF1]] : (tensor<3x3x64x256xf32> tensor<8x2x3x3x32x32xf32>) -> tensor<8x2x3x3x32x32xf32>
// CHECK: %[[BUF2:.+]] = tensor.empty() : tensor<1x8x56x56x32xf32>
// CHECK: %[[PACK2:.+]] = linalgx.pack %[[ARG2]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUF2]] : (tensor<1x56x56x256xf32> tensor<1x8x56x56x32xf32>) -> tensor<1x8x56x56x32xf32>
// CHECK: %[[GEN:.+]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%[[PACK0]], %[[PACK1]] : tensor<1x2x113x113x32xf32>, tensor<8x2x3x3x32x32xf32>) outs(%[[PACK2]] : tensor<1x8x56x56x32xf32>)
// CHECK: %[[RES:.+]] = linalgx.unpack %[[GEN]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[ARG2]] : (tensor<1x8x56x56x32xf32> tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
// CHECK: return %[[RES]] : tensor<1x56x56x256xf32>
