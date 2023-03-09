// RUN: tpp-opt %s -convert-tpp-to-xsmm -split-input-file | FileCheck %s

// CHECK-LABEL: @matmul_to_xsmm(
// CHECK-SAME: %[[ARG0:.*]]: memref<3x3xf32>, %[[ARG1:.*]]: memref<3x3xf32>, %[[ARG2:.*]]: memref<3x3xf32>)
func.func @matmul_to_xsmm(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>, %arg2: memref<3x3xf32>) {
  // CHECK: %[[DISPATCH:.*]] = xsmm.ternary.dispatch matmul [3, 3, 3, 3, 3, 3](dataType f32, isVNNI false)
  // CHECK-NEXT: xsmm.ternary matmul(dataType f32, %[[DISPATCH]], %[[ARG0]], %[[ARG1]], %[[ARG2]]) 
  tpp.matmul ins(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) out(%arg2: memref<3x3xf32>)
  return
}

// -----

// Strides are non-constant expect to fail.
func.func @tpp_matmul(%arg0: memref<12x9xf32, strided<[?, ?], offset: ?>>,
                      %arg1: memref<9x6xf32, strided<[?, ?], offset: ?>>,
                      %arg2: memref<12x6xf32, strided<[?, ?], offset: ?>>) {
  // CHECK-NOT: xsmm.ternary matmul
  tpp.matmul ins(%arg0 : memref<12x9xf32, strided<[?, ?], offset: ?>>,
                 %arg1 : memref<9x6xf32, strided<[?, ?], offset: ?>>)
             out(%arg2 : memref<12x6xf32, strided<[?, ?], offset: ?>>)
  return
}
