// RUN: tpp-opt %s -pack-matmul="block-factors=32,32,32" -canonicalize -tile-consumer-and-fuse-producers="tile-sizes=1,1" -canonicalize -rewrite-to-brgemm | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @matmul_and_relu(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  %c0 = arith.constant 0.0 : f32
  %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%0: tensor<128x128xf32>) {
    ^bb0(%out: f32):
      %2 = arith.maxf %out, %c0 : f32
      linalg.yield %2 : f32
    } -> tensor<128x128xf32>
  return %1 : tensor<128x128xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: func.func @matmul_and_relu(
// CHECK-SAME:    %[[ARG0:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:    %[[ARG1:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:    %[[ARG2:[0-9a-z]+]]: tensor<128x128xf32>) -> tensor<128x128xf32> {
// CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK: %[[BUF0:.+]] = tensor.empty() : tensor<4x4x32x32xf32>
// CHECK: %[[PACK0:.+]] = tensor.pack %[[ARG0]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUF0]] : tensor<128x128xf32> -> tensor<4x4x32x32xf32>
// CHECK: %[[BUF1:.+]] = tensor.empty() : tensor<4x4x32x32xf32>
// CHECK: %[[PACK1:.+]] = tensor.pack %[[ARG1]] outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUF1]] : tensor<128x128xf32> -> tensor<4x4x32x32xf32>
// CHECK: %[[BUF2:.+]] = tensor.empty() : tensor<4x4x32x32xf32>
// CHECK: %[[PACK2:.+]] = tensor.pack %[[ARG2]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUF2]] : tensor<128x128xf32> -> tensor<4x4x32x32xf32>
// CHECK: %[[LOOP0:.+]] = scf.for %[[ARG3:.+]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%[[ARG4:.+]] = %[[PACK2]]) -> (tensor<4x4x32x32xf32>) {
// CHECK: %[[LOOP1:.+]] = scf.for %[[ARG5:.+]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<4x4x32x32xf32>) {
// CHECK: %[[SLICE0:.+]] = tensor.extract_slice %[[PACK0]][%[[ARG3]], 0, 0, 0] [1, 4, 32, 32] [1, 1, 1, 1] : tensor<4x4x32x32xf32> to tensor<4x32x32xf32>
// CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[PACK1]][%[[ARG5]], 0, 0, 0] [1, 4, 32, 32] [1, 1, 1, 1] : tensor<4x4x32x32xf32> to tensor<4x32x32xf32>
// CHECK: %[[SLICE2:.+]] = tensor.extract_slice %[[ARG6]][%[[ARG3]], %[[ARG5]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<4x4x32x32xf32> to tensor<32x32xf32>
// CHECK: %[[MUL:.+]] = linalg.batch_reduce_matmul ins(%[[SLICE0]], %[[SLICE1]] : tensor<4x32x32xf32>, tensor<4x32x32xf32>) outs(%[[SLICE2]] : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK: %[[RELU:.+]] = linalg.generic {indexing_maps = [#[[MAP]]], iterator_types = ["parallel", "parallel"]} outs(%[[MUL]] : tensor<32x32xf32>)
