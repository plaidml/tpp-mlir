// RUN: tpp-opt %s -default-tpp-passes -split-input-file | FileCheck %s

// CHECK: func.func @entry(
// CHECK-SAME:  %[[ARG0:.+]]: memref<4x8xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<8x4xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: memref<4x4xf32>)
func.func @entry(%A: memref<4x8xf32>,
          %B: memref<8x4xf32>, %C: memref<4x4xf32>) {
  // CHECK: call @xsmm_matmul_dispatch
  // CHECK: %[[cast0:.*]] = memref.cast %[[ARG0]]
  // CHECK: %[[cast1:.*]] = memref.cast %[[ARG1]]
  // CHECK: %[[cast2:.*]] = memref.cast %[[ARG2]]
  // CHECK: call @xsmm_matmul_invoke({{.*}}%[[cast0]], %[[cast1]], %[[cast2]]
  %0 = xsmm.ternary.dispatch matmul [4, 4, 8, 8, 4, 4](dataType f32)
  xsmm.ternary matmul(dataType f32, %0, %A, %B, %C) : (i64, memref<4x8xf32>, memref<8x4xf32>, memref<4x4xf32>) -> ()

  // CHECK: return
  return
}
