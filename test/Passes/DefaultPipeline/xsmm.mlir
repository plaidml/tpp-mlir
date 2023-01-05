// RUN: tpp-opt %s -default-tpp-passes -split-input-file | FileCheck %s

// CHECK: func.func @matmul(
// CHECK-SAME:  %[[ARG0:.+]]: memref<4x8xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<8x4xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: memref<4x4xf32>)
func.func @matmul(%A: memref<4x8xf32>,
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

// -----

#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK: func.func @mlp(
// CHECK-SAME:  %[[ARG0:.+]]: memref<128x256xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<256x512xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: memref<512xf32>,
// CHECK-SAME:  %[[ARG3:.+]]: memref<128x512xf32>)
module @predict_function {
  func.func @mlp(%arg0: memref<128x256xf32>, %arg1: memref<256x512xf32>,
    %arg2: memref<512xf32>,  %arg3: memref<128x512xf32>) {

    // Identity
    // CHECK: call @xsmm_unary_dispatch
    // CHECK: %[[cast:.*]] = memref.cast %[[ARG2]]
    // CHECK: %[[cast0:.*]] = memref.cast %[[ARG3]]
    // CHECK: call @xsmm_unary_invoke({{.*}}%[[cast]], %[[cast0]]
    %0 = xsmm.unary.dispatch identity [128, 512, 512, 512](broadcast col dataType f32)
    xsmm.unary identity(dataType f32, %0, %arg2, %arg3) : (i64, memref<512xf32>, memref<128x512xf32>) -> ()

    // Matmul
    // CHECK: call @xsmm_matmul_dispatch
    // CHECK: %[[cast1:.*]] = memref.cast %[[ARG0]]
    // CHECK: %[[cast2:.*]] = memref.cast %[[ARG1]]
    // CHECK: %[[cast3:.*]] = memref.cast %[[ARG3]]
    // CHECK: call @xsmm_matmul_invoke({{.*}}%[[cast1]], %[[cast2]], %[[cast3]]
    %1 = xsmm.ternary.dispatch matmul [128, 512, 256, 256, 512, 512](dataType f32)
    xsmm.ternary matmul(dataType f32, %1, %arg0, %arg1, %arg3) : (i64, memref<128x256xf32>, memref<256x512xf32>, memref<128x512xf32>) -> ()

    // Relu
    // CHECK: call @xsmm_unary_dispatch
    // CHECK: %[[cast4:.*]] = memref.cast %[[ARG3]]
    // CHECK: call @xsmm_unary_invoke_inline({{.*}}%[[cast4]]
    %2 = xsmm.unary.dispatch relu [128, 512, 512, 512](broadcast none dataType f32)
    xsmm.unary relu(dataType f32, %2, %arg3) : (i64, memref<128x512xf32>) -> ()

    // CHECK: return
    return
  }
}
