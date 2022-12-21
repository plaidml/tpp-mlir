// RUN: tpp-opt %s -map-linalg-to-tpp \
// RUN: -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map" \
// RUN: -drop-equivalent-buffer-results -finalizing-bufferize -canonicalize \
// RUN: -convert-linalg-to-tpp | FileCheck %s

func.func @entry(%A: tensor<4x8xf32>,
          %B: tensor<8x4xf32>, %C: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %D = linalg.matmul ins(%A, %B: tensor<4x8xf32>, tensor<8x4xf32>) outs(%C: tensor<4x4xf32>) -> tensor<4x4xf32>
  return %D : tensor<4x4xf32>
}

// CHECK: func.func @entry(
// CHECK-SAME:  %[[ARG0:.+]]: memref<4x8xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<8x4xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: memref<4x4xf32>)
// CHECK: tpp.matmul ins(%[[ARG0]] : memref<4x8xf32>, %[[ARG1]] : memref<8x4xf32>) out(%[[ARG2]] : memref<4x4xf32>)
// CHECK: return
