// RUN: standalone-opt -split-input-file -map-to-brgemm %s | FileCheck %s

#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

// CHECK-LABEL: func.func @blocked_matmul(
// CHECK-SAME: %[[arg_zero:.*]]: tensor<4x16x32x32xf32>,
// CHECK-SAME: %[[arg_one:.*]]: tensor<8x16x32x32xf32>,
// CHECK-SAME: %[[arg_two:.*]]: tensor<4x8x32x32xf32>)
func.func @blocked_matmul(%arg0: tensor<4x16x32x32xf32>, %arg1: tensor<8x16x32x32xf32>, %arg2: tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> {
  // CHECK-DAG: %[[cst_zero:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[cst_four:.*]] = arith.constant 4 : index
  // CHECK-DAG: %[[cst_one:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[cst_eight:.*]] = arith.constant 8 : index
  // CHECK: %[[outer:.*]] = scf.for %[[p1:.*]] = %[[cst_zero]] to %[[cst_four]] step %[[cst_one]] iter_args(%[[init:.*]] = %[[arg_two]]) -> (tensor<4x8x32x32xf32>) {
  // CHECK: %[[inner:.*]] = scf.for %[[p2:.*]] = %[[cst_zero]] to %[[cst_eight]] step %[[cst_one]] iter_args(%[[init2:.*]] = %[[init]]) -> (tensor<4x8x32x32xf32>) {
  // CHECK: %[[sliceA:.*]] = tensor.extract_slice %[[arg_zero]][%[[p1]], 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : tensor<4x16x32x32xf32> to tensor<16x32x32xf32>
  // CHECK: %[[sliceB:.*]] = tensor.extract_slice %[[arg_one]][%[[p2]], 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : tensor<8x16x32x32xf32> to tensor<16x32x32xf32>
  // CHECK: %[[sliceC:.*]] = tensor.extract_slice %[[init2]][%[[p1]], %[[p2]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<4x8x32x32xf32> to tensor<32x32xf32>
  // CHECK: %[[mul:.*]] = linalg.reduce_batch_matmul ins(%[[sliceA]], %[[sliceB]] : tensor<16x32x32xf32>, tensor<16x32x32xf32>) outs(%[[sliceC]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[yield:.*]] = tensor.insert_slice %[[mul]] into %[[init2]][%[[p1]], %[[p2]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xf32> into tensor<4x8x32x32xf32>
  // CHECK: scf.yield %[[yield]] : tensor<4x8x32x32xf32>
  // CHECK: }
  // CHECK: scf.yield %[[inner]] : tensor<4x8x32x32xf32>
  // CHECK: }
  // CHECK: return %[[outer]] : tensor<4x8x32x32xf32>
  %1 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<4x16x32x32xf32>, tensor<8x16x32x32xf32>) outs(%arg2 : tensor<4x8x32x32xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %8 = arith.mulf %arg3, %arg4 : f32
      %9 = arith.addf %arg5, %8 : f32
      linalg.yield %9 : f32
    } -> tensor<4x8x32x32xf32>
  return %1 :  tensor<4x8x32x32xf32>
}

// -----

#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0, d1, d2)[s0] -> (d0 * 1024 + s0 + d1 * 32 + d2)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1)[s0] -> (d0 * 32 + s0 + d1)>

// CHECK-LABEL: func.func @blocked_matmul(
// CHECK-SAME: %[[arg_zero:.*]]: memref<4x16x32x32xf32>, %[[arg_one:.*]]: memref<8x16x32x32xf32>, %[[arg_two:.*]]: memref<4x8x32x32xf32>) -> memref<4x8x32x32xf32> { 
func.func @blocked_matmul(%arg0: memref<4x16x32x32xf32>, %arg1: memref<8x16x32x32xf32>, %arg2: memref<4x8x32x32xf32>) -> memref<4x8x32x32xf32> {
  // CHECK-DAG: %[[cst_zero:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[cst_four:.*]] = arith.constant 4 : index
  // CHECK-DAG: %[[cst_one:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[cst_eight:.*]] = arith.constant 8 : index
  // CHECK: scf.for %[[p1:.*]] = %[[cst_zero]] to %[[cst_four]] step %[[cst_one]] {
  // CHECK-NEXT: scf.for %[[p2:.*]] = %[[cst_zero]] to %[[cst_eight]] step %[[cst_one]] {
  // CHECK-NEXT: %[[l1:.*]] = memref.subview %[[arg_zero]][%[[p1]], 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<4x16x32x32xf32> to memref<16x32x32xf32, #[[MAP0]]>
  // CHECK-NEXT: %[[l2:.*]] = memref.subview %[[arg_one]][%[[p2]], 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<8x16x32x32xf32> to memref<16x32x32xf32, #[[MAP0]]>
  // CHECK-NEXT: %[[l3:.*]] = memref.subview %[[arg_two]][%[[p1]], %[[p2]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<4x8x32x32xf32> to memref<32x32xf32, #[[MAP1]]>
  // CHECK-NEXT: linalg.reduce_batch_matmul ins(%[[l1]], %[[l2]] : memref<16x32x32xf32, #[[MAP0]]>, memref<16x32x32xf32, #[[MAP0]]>) outs(%[[l3]] : memref<32x32xf32, #[[MAP1]]>)
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[arg_two]] : memref<4x8x32x32xf32>
  linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : memref<4x16x32x32xf32>, memref<8x16x32x32xf32>) outs(%arg2 : memref<4x8x32x32xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %8 = arith.mulf %arg3, %arg4 : f32
      %9 = arith.addf %arg5, %8 : f32
      linalg.yield %9 : f32
    }
  return %arg2 : memref<4x8x32x32xf32>
}
