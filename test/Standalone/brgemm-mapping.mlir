// RUN: tpp-opt -split-input-file -map-to-brgemm %s | FileCheck %s

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
  // CHECK: %[[mul:.*]] = linalg.batch_reduce_matmul ins(%[[sliceA]], %[[sliceB]] : tensor<16x32x32xf32>, tensor<16x32x32xf32>) outs(%[[sliceC]] : tensor<32x32xf32>) -> tensor<32x32xf32>
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

// CHECK-LABEL: func.func @blocked_matmul(
// CHECK-SAME: %[[arg_zero:.*]]: memref<4x16x32x32xf32>, %[[arg_one:.*]]: memref<8x16x32x32xf32>, %[[arg_two:.*]]: memref<4x8x32x32xf32>) -> memref<4x8x32x32xf32> { 
func.func @blocked_matmul(%arg0: memref<4x16x32x32xf32>, %arg1: memref<8x16x32x32xf32>, %arg2: memref<4x8x32x32xf32>) -> memref<4x8x32x32xf32> {
  // CHECK-DAG: %[[cst_zero:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[cst_four:.*]] = arith.constant 4 : index
  // CHECK-DAG: %[[cst_one:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[cst_eight:.*]] = arith.constant 8 : index
  // CHECK: scf.for %[[p1:.*]] = %[[cst_zero]] to %[[cst_four]] step %[[cst_one]] {
  // CHECK-NEXT: scf.for %[[p2:.*]] = %[[cst_zero]] to %[[cst_eight]] step %[[cst_one]] {
  // CHECK-NEXT: %[[l1:.*]] = memref.subview %[[arg_zero]][%[[p1]], 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<4x16x32x32xf32> to memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>
  // CHECK-NEXT: %[[l2:.*]] = memref.subview %[[arg_one]][%[[p2]], 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<8x16x32x32xf32> to memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>
  // CHECK-NEXT: %[[l3:.*]] = memref.subview %[[arg_two]][%[[p1]], %[[p2]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<4x8x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
  // CHECK-NEXT: linalg.batch_reduce_matmul ins(%[[l1]], %[[l2]] : memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>) outs(%[[l3]] : memref<32x32xf32, strided<[32, 1], offset: ?>>)
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

// -----

#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>

// CHECK-LABEL: func.func @blocked_matmul
// CHECK-SAME: %[[arg_zero:.*]]: tensor<8x32x32xf32>, %[[arg_one:.*]]: tensor<8x32x32xf32>, %[[arg_two:.*]]: tensor<32x32xf32>) -> tensor<32x32xf32> {
func.func @blocked_matmul(%arg0: tensor<8x32x32xf32>, %arg1: tensor<8x32x32xf32>, %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK: %[[r:.*]] = linalg.batch_reduce_matmul ins(%[[arg_zero]], %[[arg_one]] : tensor<8x32x32xf32>, tensor<8x32x32xf32>) outs(%[[arg_two]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  %0 = linalg.generic {indexing_maps = [#map5, #map6, #map7], iterator_types = ["reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<8x32x32xf32>, tensor<8x32x32xf32>) outs(%arg2 : tensor<32x32xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5:f32):
    %m = arith.mulf %arg3, %arg4 : f32
    %a = arith.addf %arg5, %m : f32
    linalg.yield %a : f32
  } -> tensor<32x32xf32>
  // CHECK: return %[[r]] : tensor<32x32xf32>
  return %0: tensor<32x32xf32>
}

// -----

#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4)>

// CHECK-LABEL: func.func @blocked_matmul
func.func @blocked_matmul(%arg0: tensor<4x8x32x32xf32>, %arg1: tensor<16x8x32x32xf32>, %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK-DAG: %[[zero:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[four:.*]] = arith.constant 4 : index
  // CHECK-DAG: %[[one:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[sixteen:.*]] = arith.constant 16 : index
  // CHECK: %[[outer:.*]] = scf.for %arg3 = %[[zero]] to %[[four]] step %[[one]] iter_args(%arg4 = %arg2) -> (tensor<32x32xf32>) {
  // CHECK: %[[inner:.*]] = scf.for %arg5 = %[[zero]] to %[[sixteen]] step %[[one]] iter_args(%arg6 = %arg4) -> (tensor<32x32xf32>) {
  // CHECK: %[[sliceA:.*]] = tensor.extract_slice %arg0[%arg3, 0, 0, 0] [1, 8, 32, 32] [1, 1, 1, 1] : tensor<4x8x32x32xf32> to tensor<8x32x32xf32>
  // CHECK: %[[sliceB:.*]] = tensor.extract_slice %arg1[%arg5, 0, 0, 0] [1, 8, 32, 32] [1, 1, 1, 1] : tensor<16x8x32x32xf32> to tensor<8x32x32xf32>
  // CHECK: %[[r:.*]] = linalg.batch_reduce_matmul ins(%[[sliceA]], %[[sliceB]] : tensor<8x32x32xf32>, tensor<8x32x32xf32>) outs(%arg6 : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: scf.yield %[[r]] : tensor<32x32xf32>
  // CHECK: }
  // CHECK: scf.yield %[[inner]] : tensor<32x32xf32>
  // CHECK: return %[[outer]] : tensor<32x32xf32>
  %0 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1: tensor<4x8x32x32xf32>, tensor<16x8x32x32xf32>) outs(%arg2: tensor<32x32xf32>) {
  ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
    %mul = arith.mulf %arg4, %arg5 : f32
    %add = arith.addf %arg6, %mul : f32
    linalg.yield %add : f32
  } -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// -----

#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d3, d4)>

// CHECK-LABEL: func.func @blocked_matmul
func.func @blocked_matmul(%arg0: tensor<4x8x32x32xf32>, %arg1: tensor<16x8x32x32xf32>, %arg2: tensor<16x32x32xf32>) -> tensor<16x32x32xf32> {
  // CHECK-DAG: %[[zero:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[four:.*]] = arith.constant 4 : index
  // CHECK-DAG: %[[one:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[sixteen:.*]] = arith.constant 16 : index
  // CHECK: %[[outer:.*]] = scf.for %arg3 = %[[zero]] to %[[four]] step %[[one]] iter_args(%arg4 = %arg2) -> (tensor<16x32x32xf32>) {
  // CHECK: %[[inner:.*]] = scf.for %arg5 = %[[zero]] to %[[sixteen]] step %[[one]] iter_args(%arg6 = %arg4) -> (tensor<16x32x32xf32>) {
  // CHECK: %[[sliceA:.*]] = tensor.extract_slice %arg0[%arg3, 0, 0, 0] [1, 8, 32, 32] [1, 1, 1, 1] : tensor<4x8x32x32xf32> to tensor<8x32x32xf32>
  // CHECK: %[[sliceB:.*]] = tensor.extract_slice %arg1[%arg5, 0, 0, 0] [1, 8, 32, 32] [1, 1, 1, 1] : tensor<16x8x32x32xf32> to tensor<8x32x32xf32>
  // CHECK: %[[sliceC:.*]] = tensor.extract_slice %arg6[%arg5, 0, 0] [1, 32, 32] [1, 1, 1] : tensor<16x32x32xf32> to tensor<32x32xf32>
  // CHECK: %[[r:.*]] = linalg.batch_reduce_matmul ins(%[[sliceA]], %[[sliceB]] : tensor<8x32x32xf32>, tensor<8x32x32xf32>) outs(%[[sliceC]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[full:.*]] = tensor.insert_slice %[[r]] into %arg6[%arg5, 0, 0] [1, 32, 32] [1, 1, 1] : tensor<32x32xf32> into tensor<16x32x32xf32>
  // CHECK: scf.yield %[[full]] : tensor<16x32x32xf32>
  // CHECK: }
  // CHECK: scf.yield %[[inner]] : tensor<16x32x32xf32>
  // CHECK: }
  // CHECK: return %[[outer]] : tensor<16x32x32xf32>
  %0 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1: tensor<4x8x32x32xf32>, tensor<16x8x32x32xf32>) outs(%arg2: tensor<16x32x32xf32>) {
  ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
    %mul = arith.mulf %arg4, %arg5 : f32
    %add = arith.addf %arg6, %mul : f32
    linalg.yield %add : f32
  } -> tensor<16x32x32xf32>
  return %0 : tensor<16x32x32xf32>
}
