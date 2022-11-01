// RUN: tpp-opt %s -pack-matmul="block-factors=32,32,32" -canonicalize -split-input-file | FileCheck -check-prefix=MATMUL %s
// RUN: tpp-opt %s -pack-conv2DNhwcHwcf="block-factors=32,32" -canonicalize -split-input-file | FileCheck -check-prefix=CONV %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @matmul(%arg0: tensor<128x512xf32>, 
                  %arg1: tensor<512x256xf32>, 
                  %arg2: tensor<128x256xf32>) -> tensor<128x256xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1: tensor<128x512xf32>, tensor<512x256xf32>) outs(%arg2: tensor<128x256xf32>) -> tensor<128x256xf32>
  %1 = linalg.generic {indexing_maps = [#map0], iterator_types = ["parallel", "parallel"]} outs(%0: tensor<128x256xf32>) {
    ^bb0(%arg3: f32):
      %2 = mathx.relu %arg3 : f32
      linalg.yield %2 : f32
  } -> tensor<128x256xf32>
  return %1 : tensor<128x256xf32>
}

// MATMUL-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// MATMUL-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// MATMUL-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// MATMUL-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// MATMUL: func.func @matmul(
// MATMUL-SAME:  %[[ARG0:.+]]: tensor<128x512xf32>, 
// MATMUL-SAME:  %[[ARG1:.+]]: tensor<512x256xf32>,
// MATMUL-SAME:  %[[ARG2:.+]]: tensor<128x256xf32>) -> tensor<128x256xf32> {
// MATMUL: %[[BUFF0:.+]] = tensor.empty() : tensor<4x16x32x32xf32>
// MATMUL: %[[PACK0:.+]] = linalgx.pack %[[ARG0]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUFF0]] : (tensor<128x512xf32> tensor<4x16x32x32xf32>) -> tensor<4x16x32x32xf32>
// MATMUL: %[[BUFF1:.+]] = tensor.empty() : tensor<8x16x32x32xf32>
// MATMUL: %[[PACK1:.+]] = linalgx.pack %[[ARG1]] outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUFF1]] : (tensor<512x256xf32> tensor<8x16x32x32xf32>) -> tensor<8x16x32x32xf32>
// MATMUL: %[[BUFF2:.+]] = tensor.empty() : tensor<4x8x32x32xf32>
// MATMUL: %[[PACK2:.+]] = linalgx.pack %[[ARG2]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUFF2]] : (tensor<128x256xf32> tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32>
// MATMUL: %[[VAL:.+]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%[[PACK0]], %[[PACK1]] : tensor<4x16x32x32xf32>, tensor<8x16x32x32xf32>) outs(%[[PACK2]] : tensor<4x8x32x32xf32>)
// MATMUL: %[[VAL1:.+]] = linalg.generic {indexing_maps = [#[[MAP3]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%[[VAL]] : tensor<4x8x32x32xf32>)
// MATMUL: %[[OUT:.+]] = linalgx.unpack %[[VAL1]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[ARG2]] : (tensor<4x8x32x32xf32> tensor<128x256xf32>) -> tensor<128x256xf32>
// MATMUL: return %[[OUT]] : tensor<128x256xf32>
// MATMUL: }

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @matmul(%arg0: tensor<128x512xf32>,
                  %arg1: tensor<512x256xf32>,
                  %arg2: tensor<128x256xf32>) -> tensor<128x256xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1: tensor<128x512xf32>, tensor<512x256xf32>) outs(%arg2: tensor<128x256xf32>) -> tensor<128x256xf32>
  %1 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%0: tensor<128x256xf32>) outs(%arg2: tensor<128x256xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      %2 = arith.addf %arg3, %arg4 : f32
      linalg.yield %2 : f32
  } -> tensor<128x256xf32>
  return %1 : tensor<128x256xf32>
}

// MATMUL-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// MATMUL-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// MATMUL-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// MATMUL-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// MATMUL: func.func @matmul(
// MATMUL-SAME:  %[[ARG0:.+]]: tensor<128x512xf32>,
// MATMUL-SAME:  %[[ARG1:.+]]: tensor<512x256xf32>,
// MATMUL-SAME:  %[[ARG2:.+]]: tensor<128x256xf32>) -> tensor<128x256xf32> {
// MATMUL: %[[BUFF0:.+]] = tensor.empty() : tensor<4x16x32x32xf32>
// MATMUL: %[[PACK0:.+]] = linalgx.pack %[[ARG0]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUFF0]] : (tensor<128x512xf32> tensor<4x16x32x32xf32>) -> tensor<4x16x32x32xf32>
// MATMUL: %[[BUFF1:.+]] = tensor.empty() : tensor<8x16x32x32xf32>
// MATMUL: %[[PACK1:.+]] = linalgx.pack %[[ARG1]] outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUFF1]] : (tensor<512x256xf32> tensor<8x16x32x32xf32>) -> tensor<8x16x32x32xf32>
// MATMUL: %[[BUFF2:.+]] = tensor.empty() : tensor<4x8x32x32xf32>
// MATMUL: %[[PACK2:.+]] = linalgx.pack %[[ARG2]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUFF2]] : (tensor<128x256xf32> tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32>
// MATMUL: %[[VAL:.+]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%[[PACK0]], %[[PACK1]] : tensor<4x16x32x32xf32>, tensor<8x16x32x32xf32>) outs(%[[PACK2]] : tensor<4x8x32x32xf32>)
// MATMUL: %[[BUFF2_2:.+]] = tensor.empty() : tensor<4x8x32x32xf32>
// MATMUL: %[[PACK2_2:.+]] = linalgx.pack %[[ARG2]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUFF2_2]] : (tensor<128x256xf32> tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32>
// MATMUL: %[[VAL1:.+]] = linalg.generic {indexing_maps = [#[[MAP3]], #[[MAP3]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[VAL]] : tensor<4x8x32x32xf32>) outs(%[[PACK2_2]] : tensor<4x8x32x32xf32>)
// MATMUL: %[[OUT:.+]] = linalgx.unpack %[[VAL1]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[ARG2]] : (tensor<4x8x32x32xf32> tensor<128x256xf32>) -> tensor<128x256xf32>
// MATMUL: return %[[OUT]] : tensor<128x256xf32>
// MATMUL: }

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @conv(%arg0: tensor<1x56x56x64xf32>, %arg1: tensor<1x1x64x64xf32>, %arg2: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf ins(%arg0, %arg1: tensor<1x56x56x64xf32>, tensor<1x1x64x64xf32>) outs(%arg2: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %1 = linalg.generic {indexing_maps = [#map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%0 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32):
      %13 = mathx.relu %in : f32
      linalg.yield %13 : f32
  } -> tensor<1x56x56x64xf32>
  return %1 : tensor<1x56x56x64xf32>
}

// CONV-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
// CONV-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
// CONV-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
// CONV-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CONV: func.func @conv(
// CONV-SAME: %[[ARG0:.+]]: tensor<1x56x56x64xf32>,
// CONV-SAME: %[[ARG1:.+]]: tensor<1x1x64x64xf32>,
// CONV-SAME: %[[ARG2:.+]]: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32> {
// CONV: %[[BUFF0:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CONV: %[[PACK0:.+]] = linalgx.pack %[[ARG0]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF0]] : (tensor<1x56x56x64xf32> tensor<1x2x56x56x32xf32>) -> tensor<1x2x56x56x32xf32>
// CONV: %[[BUFF1:.+]] = tensor.empty() : tensor<2x2x1x1x32x32xf32>
// CONV: %[[PACK1:.+]] = linalgx.pack %[[ARG1]] outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %[[BUFF1]] : (tensor<1x1x64x64xf32> tensor<2x2x1x1x32x32xf32>) -> tensor<2x2x1x1x32x32xf32>
// CONV: %[[BUFF2:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CONV: %[[PACK2:.+]] = linalgx.pack %[[ARG2]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF2]] : (tensor<1x56x56x64xf32> tensor<1x2x56x56x32xf32>) -> tensor<1x2x56x56x32xf32>
// CONV: %[[VAL:.+]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%[[PACK0]], %[[PACK1]] : tensor<1x2x56x56x32xf32>, tensor<2x2x1x1x32x32xf32>) outs(%[[PACK2]] : tensor<1x2x56x56x32xf32>)
// CONV: %[[VAL1:.+]] = linalg.generic {indexing_maps = [#[[MAP3]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} outs(%[[VAL]] : tensor<1x2x56x56x32xf32>)
// CONV: %[[UNPACK:.+]] = linalgx.unpack %[[VAL1]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[ARG2]] : (tensor<1x2x56x56x32xf32> tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
// CONV: return %[[UNPACK]] : tensor<1x56x56x64xf32>

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

// CONV-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
// CONV-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
// CONV-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
// CONV-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CONV: func.func @conv(
// CONV-SAME: %[[ARG0:.+]]: tensor<1x56x56x64xf32>,
// CONV-SAME: %[[ARG1:.+]]: tensor<1x1x64x64xf32>,
// CONV-SAME: %[[ARG2:.+]]: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32> {
// CONV: %[[BUFF0:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CONV: %[[PACK0:.+]] = linalgx.pack %[[ARG0]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF0]] : (tensor<1x56x56x64xf32> tensor<1x2x56x56x32xf32>) -> tensor<1x2x56x56x32xf32>
// CONV: %[[BUFF1:.+]] = tensor.empty() : tensor<2x2x1x1x32x32xf32>
// CONV: %[[PACK1:.+]] = linalgx.pack %[[ARG1]] outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %[[BUFF1]] : (tensor<1x1x64x64xf32> tensor<2x2x1x1x32x32xf32>) -> tensor<2x2x1x1x32x32xf32>
// CONV: %[[BUFF2:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CONV: %[[PACK2:.+]] = linalgx.pack %[[ARG2]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF2]] : (tensor<1x56x56x64xf32> tensor<1x2x56x56x32xf32>) -> tensor<1x2x56x56x32xf32>
// CONV: %[[VAL:.+]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%[[PACK0]], %[[PACK1]] : tensor<1x2x56x56x32xf32>, tensor<2x2x1x1x32x32xf32>) outs(%[[PACK2]] : tensor<1x2x56x56x32xf32>)
// CONV: %[[BUFF2_2:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CONV: %[[PACK2_2:.+]] = linalgx.pack %[[ARG2]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF2_2]] : (tensor<1x56x56x64xf32> tensor<1x2x56x56x32xf32>) -> tensor<1x2x56x56x32xf32>
// CONV: %[[VAL1:.+]] = linalg.generic {indexing_maps = [#[[MAP3]], #[[MAP3]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%[[VAL]] : tensor<1x2x56x56x32xf32>) outs(%[[PACK2_2]] : tensor<1x2x56x56x32xf32>)
// CONV: %[[UNPACK:.+]] = linalgx.unpack %[[VAL1]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[ARG2]] : (tensor<1x2x56x56x32xf32> tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
// CONV: return %[[UNPACK]] : tensor<1x56x56x64xf32>

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

// CONV-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
// CONV-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
// CONV-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
// CONV-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CONV-DAG: #[[MAP4:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d4)>
// CONV: func.func @conv(
// CONV-SAME: %[[ARG0:[a-zA-Z0-9]*]]: tensor<1x56x56x64xf32>,
// CONV-SAME: %[[ARG1:[a-zA-Z0-9]*]]: tensor<1x1x64x64xf32>,
// CONV-SAME: %[[ARG2:[a-zA-Z0-9]*]]: tensor<1x56x56x64xf32>,
// CONV-SAME: %[[ARG3:[a-zA-Z0-9]*]]: tensor<64xf32>) -> tensor<1x56x56x64xf32>
// CONV: %[[BUFF0:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CONV: %[[PACK0:.+]] = linalgx.pack %[[ARG0]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF0]] : (tensor<1x56x56x64xf32> tensor<1x2x56x56x32xf32>) -> tensor<1x2x56x56x32xf32>
// CONV: %[[BUFF1:.+]] = tensor.empty() : tensor<2x2x1x1x32x32xf32>
// CONV: %[[PACK1:.+]] = linalgx.pack %[[ARG1]] outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %[[BUFF1]] : (tensor<1x1x64x64xf32> tensor<2x2x1x1x32x32xf32>) -> tensor<2x2x1x1x32x32xf32>
// CONV: %[[BUFF2:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CONV: %[[PACK2:.+]] = linalgx.pack %[[ARG2]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF2]] : (tensor<1x56x56x64xf32> tensor<1x2x56x56x32xf32>) -> tensor<1x2x56x56x32xf32>
// CONV: %[[VAL:.+]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%[[PACK0]], %[[PACK1]] : tensor<1x2x56x56x32xf32>, tensor<2x2x1x1x32x32xf32>) outs(%[[PACK2]] : tensor<1x2x56x56x32xf32>)
// CONV: %[[EXPAND:.+]] = tensor.expand_shape %[[ARG3]] {{\[}}[0, 1]] : tensor<64xf32> into tensor<2x32xf32>
// CONV: %[[BUFF3:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CONV: %[[PACK3:.+]] = linalgx.pack %[[ARG2]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF3]] : (tensor<1x56x56x64xf32> tensor<1x2x56x56x32xf32>) -> tensor<1x2x56x56x32xf32>
// CONV: %[[VAL1:.+]] = linalg.generic {indexing_maps = [#[[MAP3]], #[[MAP4]], #[[MAP3]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%[[VAL]], %[[EXPAND]] : tensor<1x2x56x56x32xf32>, tensor<2x32xf32>) outs(%[[PACK3]] : tensor<1x2x56x56x32xf32>)
// CONV: %[[UNPACK:.+]] = linalgx.unpack %[[VAL1]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[ARG2]] : (tensor<1x2x56x56x32xf32> tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>

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

// CONV-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
// CONV-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
// CONV-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
// CONV-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CONV-DAG: #[[MAP4:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)> 
// CONV: func.func @conv(
// CONV-SAME: %[[ARG0:[a-zA-Z0-9]*]]: tensor<1x56x56x64xf32>,
// CONV-SAME: %[[ARG1:[a-zA-Z0-9]*]]: tensor<1x1x64x64xf32>,
// CONV-SAME: %[[ARG2:[a-zA-Z0-9]*]]: tensor<1x56x56x64xf32>,
// CONV-SAME: %[[ARG3:[a-zA-Z0-9]*]]: tensor<56x64xf32>) -> tensor<1x56x56x64xf32>
// CONV: %[[BUFF0:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CONV: %[[PACK0:.+]] = linalgx.pack %[[ARG0]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF0]] : (tensor<1x56x56x64xf32> tensor<1x2x56x56x32xf32>) -> tensor<1x2x56x56x32xf32>
// CONV: %[[BUFF1:.+]] = tensor.empty() : tensor<2x2x1x1x32x32xf32>
// CONV: %[[PACK1:.+]] = linalgx.pack %[[ARG1]] outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %[[BUFF1]] : (tensor<1x1x64x64xf32> tensor<2x2x1x1x32x32xf32>) -> tensor<2x2x1x1x32x32xf32>
// CONV: %[[BUFF2:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CONV: %[[PACK2:.+]] = linalgx.pack %[[ARG2]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF2]] : (tensor<1x56x56x64xf32> tensor<1x2x56x56x32xf32>) -> tensor<1x2x56x56x32xf32>
// CONV: %[[VAL:.+]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%[[PACK0]], %[[PACK1]] : tensor<1x2x56x56x32xf32>, tensor<2x2x1x1x32x32xf32>) outs(%[[PACK2]] : tensor<1x2x56x56x32xf32>)
// CONV: %[[BUFF3:.+]] = tensor.empty() : tensor<2x56x32xf32>
// CONV: %[[PACK3:.+]] = linalgx.pack %[[ARG3]] outer_dims_perm = [1, 0] inner_dims_pos = [1] inner_tiles = [32] into %[[BUFF3]] : (tensor<56x64xf32> tensor<2x56x32xf32>) -> tensor<2x56x32xf32>
// CONV: %[[BUFF4:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CONV: %[[PACK4:.+]] = linalgx.pack %[[ARG2]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF4]] : (tensor<1x56x56x64xf32> tensor<1x2x56x56x32xf32>) -> tensor<1x2x56x56x32xf32>
// CONV: %[[VAL1:.+]] = linalg.generic {indexing_maps = [#[[MAP3]], #[[MAP4]], #[[MAP3]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%[[VAL]], %[[PACK3]] : tensor<1x2x56x56x32xf32>, tensor<2x56x32xf32>) outs(%[[PACK4]] : tensor<1x2x56x56x32xf32>)
// CONV: %[[UNPACK:.+]] = linalgx.unpack %[[VAL1]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[ARG2]] : (tensor<1x2x56x56x32xf32> tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32> 
