// RUN: tpp-opt %s -pack-matmul -split-input-file | FileCheck %s

func.func @block_linalg_matmul(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32> {
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// CHECK-DAG: #[[MAP3:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// CHECK-DAG: #[[MAP4:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// CHECK-DAG: #[[MAP5:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

// CHECK-LABEL: func @block_linalg_matmul(
// CHECK-SAME:    %[[ARG0:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:    %[[ARG1:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:    %[[ARG2:[0-9a-z]+]]: tensor<128x128xf32>) -> tensor<128x128xf32> {
// CHECK: %[[BUF0:.+]] = tensor.empty() : tensor<4x4x32x32xf32>
// CHECK: %[[PACK0:.+]] = tensor.pack %[[ARG0]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUF0]] : tensor<128x128xf32> -> tensor<4x4x32x32xf32>
// CHECK: %[[BUF1:.*]] = tensor.empty() : tensor<4x4x32x32xf32>
// CHECK: %[[PACK1:.+]] = tensor.pack %[[ARG1]] outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUF1]] : tensor<128x128xf32> -> tensor<4x4x32x32xf32>
// CHECK: %[[BUF2:.+]] = tensor.empty() : tensor<4x4x32x32xf32>
// CHECK: %[[PACK2:.+]] = tensor.pack %[[ARG2]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUF2]] : tensor<128x128xf32> -> tensor<4x4x32x32xf32>
// CHECK: %[[VAL:.+]] = linalg.generic {indexing_maps = [#[[MAP3]], #[[MAP4]], #[[MAP5]]], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%[[PACK0]], %[[PACK1]] : tensor<4x4x32x32xf32>, tensor<4x4x32x32xf32>) outs(%[[PACK2]] : tensor<4x4x32x32xf32>)
// CHECK: %[[OUT:.+]] = tensor.unpack %[[VAL]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[ARG2]] : tensor<4x4x32x32xf32> -> tensor<128x128xf32>
// CHECK: return %[[OUT]] : tensor<128x128xf32>
