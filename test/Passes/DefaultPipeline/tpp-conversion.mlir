// RUN: tpp-opt %s -tpp-conversion -split-input-file | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @tpp_sequence(%arg0: tensor<3x5x4xf32>, %arg1: tensor<3x4x5xf32>,
               %arg2: tensor<5x5xf32>, %arg3: tensor<5x5xf32>) -> tensor<5x5xf32> {  
  %0 = linalg.batch_reduce_matmul ins(%arg0, %arg1: tensor<3x5x4xf32>, tensor<3x4x5xf32>)
                                  outs(%arg2: tensor<5x5xf32>) -> tensor<5x5xf32>
  %c0 = arith.constant 0.0 : f32
  %1 = linalg.generic {
    indexing_maps = [#map],
    iterator_types = ["parallel", "parallel"]}
    outs(%arg2 : tensor<5x5xf32>) {
      ^bb0(%arg14: f32):
        %13 = arith.maximumf %arg14, %c0: f32
        linalg.yield %13 : f32
  } -> tensor<5x5xf32>
  %2 = linalg.matmul ins(%1, %0: tensor<5x5xf32>, tensor<5x5xf32>)
                     outs(%arg2: tensor<5x5xf32>) -> tensor<5x5xf32>
  return %2 : tensor<5x5xf32>
}

// CHECK-LABEL: func.func @tpp_sequence(
// CHECK-NOT: linalg.batch_reduce_matmul
// CHECK: tpp.brgemm
// CHECK-NOT: linalg.generic
// CHECK: tpp.relu
// CHECK-NOT: linalg.matmul
// CHECK: tpp.gemm

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (0, d1)>

func.func @linalg_dialect_expect_fused_brgemm(
    %arg0: tensor<1x3x2xf32>, %arg1: tensor<1x2x3xf32>, 
    %arg2: tensor<3x3xf32>, %bias: tensor<1x3xf32>) -> tensor<3x3xf32> {
  %0 = linalg.batch_reduce_matmul ins(%arg0, %arg1: tensor<1x3x2xf32>, tensor<1x2x3xf32>)
                                  outs(%arg2: tensor<3x3xf32>) -> tensor<3x3xf32>
  %empty = tensor.empty() : tensor<3x3xf32>
  %1 = linalg.generic {
    indexing_maps = [#map, #map1, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%0, %bias: tensor<3x3xf32>, tensor<1x3xf32>)
    outs(%empty: tensor<3x3xf32>) {
      ^bb0(%in: f32, %in_1: f32, %out: f32):
        %add = arith.addf %in, %in_1 : f32
        linalg.yield %add : f32
    } -> tensor<3x3xf32>
  %c0 = arith.constant 0.0 : f32
  %2 = linalg.generic {
    indexing_maps = [#map],
    iterator_types = ["parallel", "parallel"]}
    outs(%1: tensor<3x3xf32>) {
      ^bb0(%out: f32):
        %relu = arith.maximumf %out, %c0 : f32
        linalg.yield %relu : f32
  } -> tensor<3x3xf32>
  return %2: tensor<3x3xf32>
}

// CHECK-LABEL: linalg_dialect_expect_fused_brgemm
// CHECK-SAME:  %[[ARG0:.+]]: tensor<1x3x2xf32>, %[[ARG1:.+]]: tensor<1x2x3xf32>, 
// CHECK-SAME:  %[[ARG2:.+]]: tensor<3x3xf32>, %[[ARG3:.+]]: tensor<1x3xf32>
// CHECK: %{{.+}} = tpp.fused_brgemm [unary = relu, binary = add] 
// CHECK-SAME:  (%[[ARG0]] : tensor<1x3x2xf32>, %[[ARG1]] : tensor<1x2x3xf32>, %[[ARG2]] : tensor<3x3xf32>, %[[ARG3]] : tensor<1x3xf32>) -> (tensor<3x3xf32>)

