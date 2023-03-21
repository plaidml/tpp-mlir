// RUN: tpp-opt %s -convert-tpp-to-xsmm -split-input-file | FileCheck %s

// CHECK-LABEL: @brgemm_to_xsmm(
// CHECK-SAME:  %[[ARG0:.+]]: memref<3x5x4xf32>, %[[ARG1:.+]]: memref<3x4x5xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: memref<5x5xf32>)
func.func @brgemm_to_xsmm(%arg0: memref<3x5x4xf32>, %arg1: memref<3x4x5xf32>,
                          %arg2: memref<5x5xf32>) -> memref<5x5xf32> {
  // CHECK: %[[BATCH:.+]] = arith.constant 3 : i64
  // CHECK-NEXT: %[[DISPATCH:.+]] = xsmm.ternary.dispatch brgemm [5, 5, 4, 4, 5, 5](dataType f32, isVNNI false)
  // CHECK-NEXT: xsmm.ternary brgemm(dataType f32, %[[DISPATCH]], %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[BATCH]])
  tpp.brgemm ins(%arg0: memref<3x5x4xf32>, %arg1: memref<3x4x5xf32>)
             outs(%arg2: memref<5x5xf32>)
  return %arg2: memref<5x5xf32>
}
