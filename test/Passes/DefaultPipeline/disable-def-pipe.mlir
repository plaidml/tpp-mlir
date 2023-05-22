// RUN: tpp-opt %s -disable-def-pipe -default-tpp-passes | FileCheck %s

func.func @matmul(%A: tensor<4x8xf32>,
          %B: tensor<8x4xf32>, %C: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %D = linalg.matmul ins(%A, %B: tensor<4x8xf32>, tensor<8x4xf32>) outs(%C: tensor<4x4xf32>) -> tensor<4x4xf32>
  return %D : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @matmul(
// CHECK-NOT: {{.*}}call @xsmm_
// CHECK: linalg.matmul
