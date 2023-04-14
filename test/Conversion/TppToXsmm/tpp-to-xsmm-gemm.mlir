// RUN: tpp-opt %s -convert-tpp-to-xsmm -split-input-file | FileCheck %s

// CHECK-LABEL: @matmul_to_xsmm(
// CHECK-SAME: %[[ARG0:.*]]: memref<3x3xf32>, %[[ARG1:.*]]: memref<3x3xf32>, %[[ARG2:.*]]: memref<3x3xf32>)
func.func @matmul_to_xsmm(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>, %arg2: memref<3x3xf32>) {
  // CHECK: %[[DISPATCH:.*]] = xsmm.matmul.dispatch [3, 3, 3, 3, 3, 3] flags = (none) data_type = f32
  // CHECK-NEXT: xsmm.matmul(dataType f32, %[[DISPATCH]], %[[ARG0]], %[[ARG1]], %[[ARG2]]) 
  tpp.matmul ins(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>, %arg2: memref<3x3xf32>) 
             outs(%arg2: memref<3x3xf32>)
  return
}
