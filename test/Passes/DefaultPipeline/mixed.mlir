// RUN: tpp-opt %s -default-tpp-passes -split-input-file | FileCheck %s

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
module @predict_function  {
  func.func @mlp(%arg0: memref<128x256xf32>, %arg1: memref<256x512xf32>,
    %arg2: memref<512xf32>,  %arg3: memref<128x512xf32>) {

    // Identity
    // CHECK: call @xsmm_unary_dispatch
    // CHECK: %[[cast:.*]] = memref.cast %[[ARG2]]
    // CHECK: %[[cast0:.*]] = memref.cast %[[ARG3]]
    // CHECK: call @xsmm_unary_invoke({{.*}}%[[cast]], %[[cast0]]
    tpp.identity ins(%arg2 : memref<512xf32>) out(%arg3 : memref<128x512xf32>)

    // Matmul
    // CHECK: call @xsmm_matmul_dispatch
    // CHECK: %[[cast1:.*]] = memref.cast %[[ARG0]]
    // CHECK: %[[cast2:.*]] = memref.cast %[[ARG1]]
    // CHECK: call @xsmm_matmul_invoke({{.*}}%[[cast1]], %[[cast2]], %[[cast0]]
    linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : memref<128x256xf32>, memref<256x512xf32>) outs(%arg3 : memref<128x512xf32>) attrs =  {iterator_ranges = [128, 512, 256]} {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.mulf %in, %in_0 : f32
      %1 = arith.addf %out, %0 : f32
      linalg.yield %1 : f32
    }

    // Relu
    // CHECK: call @xsmm_unary_dispatch
    // CHECK: call @xsmm_unary_invoke_inline({{.*}}%[[cast0]], %[[cast0]]
    %2 = xsmm.unary.dispatch relu [128, 512, 512, 512](broadcast none dataType f32)
    xsmm.unary relu(dataType f32, %2, %arg3, %arg3) : (i64, memref<128x512xf32>, memref<128x512xf32>) -> ()

    // CHECK: return
    return
  }
}

// -----

// CHECK: func.func @buffer_dealloc(
// CHECK-SAME:  %[[ARG0:.+]]: memref<4x8xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<8x4xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: memref<4x4xf32>)
func.func @buffer_dealloc(%A: memref<4x8xf32>,
          %B: memref<8x4xf32>, %C: memref<4x4xf32>) {
  // CHECK: %[[alloc:.*]] = memref.alloc
  %0 = memref.alloc() : memref<4x4xf32>

  // CHECK: call @xsmm_matmul_dispatch
  // CHECK: %[[cast0:.*]] = memref.cast %[[ARG0]]
  // CHECK: %[[cast1:.*]] = memref.cast %[[ARG1]]
  // CHECK: %[[cast2:.*]] = memref.cast %[[alloc]]
  // CHECK: call @xsmm_matmul_invoke({{.*}}%[[cast0]], %[[cast1]], %[[cast2]]
  linalg.matmul ins(%A, %B : memref<4x8xf32>, memref<8x4xf32>) outs(%0 : memref<4x4xf32>)

  // CHECK: memref.copy
  memref.copy %0, %C : memref<4x4xf32> to memref<4x4xf32>

  // CHECK: memref.dealloc %[[alloc]]
  // CHECK: return
  return
}

// -----

// CHECK: func.func @buffer_no_dealloc(
// CHECK-SAME:  %[[ARG0:.+]]: memref<4x8xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<8x4xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: memref<4x4xf32>)
func.func @buffer_no_dealloc(%A: memref<4x8xf32>,
          %B: memref<8x4xf32>, %C: memref<4x4xf32>) -> memref<4x4xf32> {
  // CHECK: %[[alloc:.*]] = memref.alloc
  %0 = memref.alloc() : memref<4x4xf32>

  // CHECK: call @xsmm_matmul_dispatch
  // CHECK: %[[cast0:.*]] = memref.cast %[[ARG0]]
  // CHECK: %[[cast1:.*]] = memref.cast %[[ARG1]]
  // CHECK: %[[cast2:.*]] = memref.cast %[[alloc]]
  // CHECK: call @xsmm_matmul_invoke({{.*}}%[[cast0]], %[[cast1]], %[[cast2]]
  linalg.matmul ins(%A, %B : memref<4x8xf32>, memref<8x4xf32>) outs(%0 : memref<4x4xf32>)

  // CHECK: memref.copy
  memref.copy %0, %C : memref<4x4xf32> to memref<4x4xf32>

  // CHECK-NOT: memref.dealloc %[[alloc]]
  // CHECK: return
  return %0 : memref<4x4xf32>
}
