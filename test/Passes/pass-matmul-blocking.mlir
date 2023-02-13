// RUN: tpp-opt %s -pack-matmul="block-factors=32,32,32" -split-input-file | FileCheck %s

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
// CHECK: }

// -----

// We don't expect to block as the blocking factor do not create full tiles.
func.func @block_linalg_matmul(
  %arg0: tensor<5x6xf32>, %arg1: tensor<6x5xf32>, %arg2: tensor<5x5xf32>)
    -> tensor<5x5xf32> {
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<5x6xf32>, tensor<6x5xf32>)
                     outs(%arg2: tensor<5x5xf32>)
    -> tensor<5x5xf32>
  return %0 : tensor<5x5xf32>
}

// CHECK-LABEL: func.func @block_linalg_matmul(
// CHECK-SAME:  %[[ARG0:[0-9a-z]+]]: tensor<5x6xf32>,
// CHECK-SAME:  %[[ARG1:[0-9a-z]+]]: tensor<6x5xf32>,
// CHECK-SAME:  %[[ARG2:[0-9a-z]+]]: tensor<5x5xf32>) -> tensor<5x5xf32> {
// CHECK: %{{.+}} = linalg.matmul 
// CHECK-SAME:  ins(%[[ARG0]], %[[ARG1]]
// CHECK-SAME:  outs(%[[ARG2]]

// -----

func.func @block_linalg_matmul(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = linalg.fill ins(%cst : f32) outs(%arg2: tensor<128x128xf32>) -> tensor<128x128xf32>
  %1 = linalg.matmul ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>) 
                      outs(%0: tensor<128x128xf32>) -> tensor<128x128xf32>
  return %1 : tensor<128x128xf32>
}

// CHECK-LABEL: func @block_linalg_matmul(
// CHECK-SAME:    %[[ARG0:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:    %[[ARG1:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:    %[[ARG2:[0-9a-z]+]]: tensor<128x128xf32>) -> tensor<128x128xf32> {
// CHECK: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[FILL:.+]] = linalg.fill ins(%[[CST]] : f32) 
// CHECK-SAME:  outs(%[[ARG2]] : tensor<128x128xf32>) -> tensor<128x128xf32>
// CHECK: %[[EMPTY_ARG0:.+]] = tensor.empty() : tensor<4x4x32x32xf32>
// CHECK: %[[PACK_ARG0:.+]] = tensor.pack %[[ARG0]] 
// CHECK-SAME:  inner_dims_pos = [0, 1] inner_tiles = [32, 32] 
// CHECK-SAME:  into %[[EMPTY_ARG0]] : tensor<128x128xf32> -> tensor<4x4x32x32xf32>
// CHECK: %[[EMPTY_ARG1:.+]] = tensor.empty() : tensor<4x4x32x32xf32>
// CHECK: %[[PACK_ARG1:.+]] = tensor.pack %[[ARG1]] 
// CHECK-SAME:  outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] 
// CHECK-SAME:  into %[[EMPTY_ARG1]] : tensor<128x128xf32> -> tensor<4x4x32x32xf32>
// CHECK: %[[EMPTY_FILL:.+]] = tensor.empty() : tensor<4x4x32x32xf32>
// CHECK: %[[PACK_FILL:.+]] = tensor.pack %[[FILL]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] 
// CHECK-SAME:  into %[[EMPTY_FILL]] : tensor<128x128xf32> -> tensor<4x4x32x32xf32>
// CHECK: %[[RES:.+]] = linalg.generic
// CHECK: %{{.+}} = tensor.unpack %[[RES]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] 
// CHECK-SAME:  into %[[ARG2]] : tensor<4x4x32x32xf32> -> tensor<128x128xf32>
