// RUN: tpp-opt %s -split-input-file -convert-linalg-to-tpp | FileCheck %s

func.func @brgemm_lowering(%arg0: tensor<3x5x4xf32>, %arg1: tensor<3x4x5xf32>,
                          %arg2: tensor<5x5xf32>) -> tensor<5x5xf32> {
  %0 = linalg.batch_reduce_matmul ins(%arg0, %arg1: tensor<3x5x4xf32>, tensor<3x4x5xf32>)
                                  outs(%arg2: tensor<5x5xf32>) -> tensor<5x5xf32>
  return %0 : tensor<5x5xf32>
}

// CHECK-LABEL: brgemm_lowering
// CHECK-SAME: %[[ARG0:.+]]: tensor<3x5x4xf32>, %[[ARG1:.+]]: tensor<3x4x5xf32>, %[[ARG2:.+]]: tensor<5x5xf32>
// CHECK: %{{.+}} = tpp.brgemm 
// CHECK-SAME:  (%[[ARG0]] : tensor<3x5x4xf32>, %[[ARG1]] : tensor<3x4x5xf32>, %[[ARG2]] : tensor<5x5xf32>) 
// CHECK-SAME:  -> (tensor<5x5xf32>)

// -----

func.func @matmul_lowering(%arg0: tensor<8x9xf32>,
                           %arg1: tensor<9x8xf32>, %arg2: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1: tensor<8x9xf32>, tensor<9x8xf32>)
                     outs(%arg2: tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: matmul_lowering
// CHECK-SAME: %[[ARG0:.+]]: tensor<8x9xf32>, %[[ARG1:.+]]: tensor<9x8xf32>, %[[ARG2:.+]]: tensor<8x8xf32>
// CHECK: %{{.+}} = tpp.matmul
// CHECK-SAME: (%[[ARG0]] : tensor<8x9xf32>, %[[ARG1]] : tensor<9x8xf32>, %[[ARG2]] : tensor<8x8xf32>) 
// CHECK-SAME:  -> (tensor<8x8xf32>)
