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
  linalg.matmul ins(%A, %B : memref<4x8xf32>, memref<8x4xf32>) outs(%C : memref<4x4xf32>)

  // CHECK: return
  return
}

// -----

// Conv2D weights
memref.global "private" constant @__constant_2048x512xf32 : memref<2048x512xf32> = dense<0.00332225906> {alignment = 128 : i64}

// CHECK-LABEL: @conv2d_1x1(
// CHECK-SAME: %[[arg:.*]]: memref<1x7x7x2048xf32>) -> memref<1x7x7x512xf32> {
func.func @conv2d_1x1(%arg0: memref<1x7x7x2048xf32>) -> memref<1x7x7x512xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c7 = arith.constant 7 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = memref.get_global @__constant_2048x512xf32 : memref<2048x512xf32>

  // 1x1 Conv2D
  // CHECK: call @xsmm_matmul_dispatch
  // CHECK: %[[cast:.*]] = memref.cast
  // CHECK: %[[cast1:.*]] = memref.cast
  // CHECK: %[[cast2:.*]] = memref.cast
  // CHECK: call @xsmm_matmul_invoke({{.*}}%[[cast]], %[[cast1]], %[[cast2]]
  %alloc = memref.alloc() {alignment = 128 : i64} : memref<1x7x7x512xf32>
  linalg.fill ins(%cst : f32) outs(%alloc : memref<1x7x7x512xf32>)
  scf.for %arg1 = %c0 to %c7 step %c1 {
    %subview = memref.subview %arg0[0, %arg1, 0, 0] [1, 1, 7, 2048] [1, 1, 1, 1] : memref<1x7x7x2048xf32> to memref<7x2048xf32, strided<[2048, 1], offset: ?>>
    %subview_0 = memref.subview %alloc[0, %arg1, 0, 0] [1, 1, 7, 512] [1, 1, 1, 1] : memref<1x7x7x512xf32> to memref<7x512xf32, strided<[512, 1], offset: ?>>
    linalg.matmul ins(%subview, %0 : memref<7x2048xf32, strided<[2048, 1], offset: ?>>, memref<2048x512xf32>) outs(%subview_0 : memref<7x512xf32, strided<[512, 1], offset: ?>>)
  }

  // CHECK: return {{.*}} : memref<1x7x7x512xf32>
  return %alloc : memref<1x7x7x512xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0 + d1)>

// CHECK-LABEL: @conv2d_1x1_decomposed(
// CHECK-SAME: %[[arg:.*]]: memref<1x7x7x2048xf32>) -> memref<1x7x7x512xf32> {
func.func @conv2d_1x1_decomposed(
      %arg0 : tensor<1x7x7x2048xf32>) -> tensor<1x7x7x512xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c7 = arith.constant 7 : index

  // Conv2D weights
  %cst = arith.constant dense<0.00332225906> : tensor<2048x512xf32>

  // 1x1 Conv2D
  // CHECK: call @xsmm_matmul_dispatch
  // CHECK: %[[cast:.*]] = memref.cast
  // CHECK: %[[cast1:.*]] = memref.cast
  // CHECK: %[[cast2:.*]] = memref.cast
  // CHECK: call @xsmm_matmul_invoke({{.*}}%[[cast]], %[[cast1]], %[[cast2]]
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x7x7x512xf32>
  %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
  %2 = scf.for %arg1 = %c0 to %c1 step %c1 iter_args(%arg2 = %1) -> (tensor<1x7x7x512xf32>) {
    %3 = scf.for %arg3 = %c0 to %c7 step %c1 iter_args(%arg4 = %arg2) -> (tensor<1x7x7x512xf32>) {
      %4 = scf.for %arg5 = %c0 to %c1 step %c1 iter_args(%arg6 = %arg4) -> (tensor<1x7x7x512xf32>) {
        %5 = scf.for %arg7 = %c0 to %c1 step %c1 iter_args(%arg8 = %arg6) -> (tensor<1x7x7x512xf32>) {
          %6 = affine.apply #map(%arg3, %arg5)
          %extracted_slice = tensor.extract_slice %arg0[%arg1, %6, %arg7, 0] [1, 1, 7, 2048] [1, 1, 1, 1] : tensor<1x7x7x2048xf32> to tensor<7x2048xf32>
          %extracted_slice_1 = tensor.extract_slice %arg8[%arg1, %arg3, 0, 0] [1, 1, 7, 512] [1, 1, 1, 1] : tensor<1x7x7x512xf32> to tensor<7x512xf32>
          %7 = linalg.matmul ins(%extracted_slice, %cst : tensor<7x2048xf32>, tensor<2048x512xf32>) outs(%extracted_slice_1 : tensor<7x512xf32>) -> tensor<7x512xf32>
          %inserted_slice = tensor.insert_slice %7 into %arg8[%arg1, %arg3, 0, 0] [1, 1, 7, 512] [1, 1, 1, 1] : tensor<7x512xf32> into tensor<1x7x7x512xf32>
          scf.yield %inserted_slice : tensor<1x7x7x512xf32>
        }
        scf.yield %5 : tensor<1x7x7x512xf32>
      }
      scf.yield %4 : tensor<1x7x7x512xf32>
    }
    scf.yield %3 : tensor<1x7x7x512xf32>
  }

  // CHECK: return {{.*}} : memref<1x7x7x512xf32>
  return %2 : tensor<1x7x7x512xf32>
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
module @predict_function  {
  func.func @mlp(%arg0: memref<128x256xf32>, %arg1: memref<256x512xf32>,
    %arg2: memref<512xf32>,  %arg3: memref<128x512xf32>) {

    // Identity
    // CHECK: call @xsmm_unary_dispatch
    // CHECK: %[[cast:.*]] = memref.cast %[[ARG2]]
    // CHECK: %[[cast0:.*]] = memref.cast %[[ARG3]]
    // CHECK: call @xsmm_unary_invoke({{.*}}%[[cast]], %[[cast0]]
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg2 : memref<512xf32>) outs(%arg3 : memref<128x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }

    // Matmul
    // CHECK: call @xsmm_matmul_dispatch
    // CHECK: %[[cast1:.*]] = memref.cast %[[ARG0]]
    // CHECK: %[[cast2:.*]] = memref.cast %[[ARG1]]
    // CHECK: %[[cast3:.*]] = memref.cast %[[ARG3]]
    // CHECK: call @xsmm_matmul_invoke({{.*}}%[[cast1]], %[[cast2]], %[[cast3]]
    linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : memref<128x256xf32>, memref<256x512xf32>) outs(%arg3 : memref<128x512xf32>) attrs =  {iterator_ranges = [128, 512, 256]} {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.mulf %in, %in_0 : f32
      %1 = arith.addf %out, %0 : f32
      linalg.yield %1 : f32
    }

    // Relu
    // CHECK: call @xsmm_unary_dispatch
    // CHECK: %[[cast4:.*]] = memref.cast %[[ARG3]]
    // CHECK: call @xsmm_unary_invoke_inline({{.*}}%[[cast4]]
    %cst = arith.constant 0.000000e+00 : f32
    linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%arg3 : memref<128x512xf32>) {
    ^bb0(%out: f32):
      %0 = arith.maxf %out, %cst : f32
      linalg.yield %0 : f32
    }

    // CHECK: return
    return
  }
}
