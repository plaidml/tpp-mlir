// RUN: tpp-opt %s -def-heap-to-stack=1 -default-tpp-passes -split-input-file | FileCheck %s

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
    tpp.identity ins(%arg2 : memref<512xf32>) outs(%arg3 : memref<128x512xf32>)

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
    // CHECK: call @xsmm_unary_invoke({{.*}}%[[cast0]], %[[cast0]]
    %2 = xsmm.unary.dispatch relu [128, 512, 512, 512](broadcast none dataType f32)
    xsmm.unary relu(dataType f32, %2, %arg3, %arg3) : (i64, memref<128x512xf32>, memref<128x512xf32>) -> ()

    return
  }
}

// -----

// CHECK: func.func @buffer_dealloc(
// CHECK-SAME:  %[[ARG0:.+]]: memref<512x128xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<128x512xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: memref<512x512xf32>)
func.func @buffer_dealloc(%A: memref<512x128xf32>,
          %B: memref<128x512xf32>, %C: memref<512x512xf32>) {
  // CHECK: %[[alloc:.*]] = memref.alloc
  %0 = memref.alloc() : memref<512x512xf32>

  // CHECK: call @xsmm_matmul_dispatch
  // CHECK: %[[cast0:.*]] = memref.cast %[[ARG0]]
  // CHECK: %[[cast1:.*]] = memref.cast %[[ARG1]]
  // CHECK: %[[cast2:.*]] = memref.cast %[[alloc]]
  // CHECK: call @xsmm_matmul_invoke({{.*}}%[[cast0]], %[[cast1]], %[[cast2]]
  linalg.matmul ins(%A, %B : memref<512x128xf32>, memref<128x512xf32>) outs(%0 : memref<512x512xf32>)

  // CHECK: memref.copy
  memref.copy %0, %C : memref<512x512xf32> to memref<512x512xf32>

  // CHECK: memref.dealloc %[[alloc]]
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
  return %0 : memref<4x4xf32>
}

// -----

// CHECK: func.func @heap_to_stack(
// CHECK-SAME:  %[[ARG0:.+]]: memref<4x8xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<8x4xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: memref<4x4xf32>)
func.func @heap_to_stack(%A: memref<4x8xf32>,
          %B: memref<8x4xf32>, %C: memref<4x4xf32>) {
  // CHECK: %[[alloc:.*]] = memref.alloca
  %0 = memref.alloc() : memref<4x4xf32>

  // CHECK: call @xsmm_matmul_dispatch
  // CHECK: %[[cast0:.*]] = memref.cast %[[ARG0]]
  // CHECK: %[[cast1:.*]] = memref.cast %[[ARG1]]
  // CHECK: %[[cast2:.*]] = memref.cast %[[alloc]]
  // CHECK: call @xsmm_matmul_invoke({{.*}}%[[cast0]], %[[cast1]], %[[cast2]]
  linalg.matmul ins(%A, %B : memref<4x8xf32>, memref<8x4xf32>) outs(%0 : memref<4x4xf32>)

  // CHECK: memref.copy
  memref.copy %0, %C : memref<4x4xf32> to memref<4x4xf32>

  // CHECK-NOT: memref.dealloc
  return
}

// -----

// CHECK-LABEL: func.func @tensor_forall
func.func @tensor_forall(%arg0: tensor<32x32xbf16>) -> tensor<8x112x32x32xbf16> {
  %c8 = arith.constant 8 : index
  %c112 = arith.constant 112 : index
  %0 = tensor.empty() : tensor<8x112x32x32xbf16>
  // CHECK-NOT: scf.forall
  // CHECK: scf.parallel
  %1 = scf.forall (%i, %j) in (%c8, %c112) shared_outs(%k = %0) -> (tensor<8x112x32x32xbf16>) {
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %arg0 into %k[%i, %j, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xbf16> into tensor<8x112x32x32xbf16>
    }
  }
  return %1 : tensor<8x112x32x32xbf16>
}
