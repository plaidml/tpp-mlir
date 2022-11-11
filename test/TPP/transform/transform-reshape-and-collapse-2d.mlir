// RUN: tpp-opt %s -transform-dialect-interpreter -canonicalize -split-input-file | FileCheck %s

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
    %1 = transform.structured.reshape_2d in %0
    %2 = transform.structured.collapse_to_2d in %1
}

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @relu(%arg0: memref<64x32x32xf32>) -> memref<64x32x32xf32> {
  %c0 = arith.constant 0.0 : f32
  linalg.generic {
    indexing_maps = [#map], 
    iterator_types = ["parallel", "parallel", "parallel"]} 
    outs(%arg0 : memref<64x32x32xf32>) {
      ^bb0(%arg1: f32):
        %1 = arith.maxf %arg1, %c0: f32
        linalg.yield %1 : f32
  }
  return %arg0 : memref<64x32x32xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: func.func @relu(
// CHECK-SAME:  %[[ARG0:.+]]: memref<64x32x32xf32>) -> memref<64x32x32xf32> {
// CHECK-DAG: %[[CST0:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C64:.+]] = arith.constant 64 : index
// CHECK: scf.for %[[ARG1:.+]] = %[[C0]] to %[[C64]] step %[[C1]] {
// CHECK: %[[SUB:.+]] = memref.subview %[[ARG0]][%[[ARG1]], 0, 0] [1, 32, 32] [1, 1, 1] : memref<64x32x32xf32> to memref<1x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK: %[[COLLAPSE:.+]] = memref.collapse_shape %[[SUB]] {{\[}}[0, 1], [2]] : memref<1x32x32xf32, strided<[1024, 32, 1], offset: ?>> into memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK: linalg.generic {indexing_maps = [#[[MAP]]], iterator_types = ["parallel", "parallel"]} outs(%[[COLLAPSE]] : memref<32x32xf32, strided<[32, 1], offset: ?>>)
// CHECK: return %[[ARG0]] : memref<64x32x32xf32>
// CHECK: }

// -----

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
    %1 = transform.structured.reshape_2d in %0
    %2 = transform.structured.collapse_to_2d in %1
}

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @additions(%arg0: memref<2x5x6xf32>, %arg1: memref<2x5x6xf32>) -> memref<2x5x6xf32> {
  linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%arg0: memref<2x5x6xf32>) outs(%arg1: memref<2x5x6xf32>) {
      ^bb0(%arg2: f32, %arg3: f32):
        %1 = arith.addf %arg2, %arg3 : f32
        linalg.yield %1 : f32
  }
  
  linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%arg0: memref<2x5x6xf32>) outs(%arg1: memref<2x5x6xf32>) {
      ^bb0(%arg2: f32, %arg3: f32):
        %1 = arith.addf %arg2, %arg3 : f32
        linalg.yield %1 : f32
  } 
  return %arg1: memref<2x5x6xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: func.func @additions(
// CHECK:  %[[ARG0:.+]]: memref<2x5x6xf32>, %[[ARG1:.+]]: memref<2x5x6xf32>) -> memref<2x5x6xf32> {
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CEHCK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index

// CHECK: scf.for %[[ARG2:.+]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK: %[[SUB:.+]] = memref.subview %[[ARG0]][%[[ARG2]], 0, 0] [1, 5, 6] [1, 1, 1] : memref<2x5x6xf32> to memref<1x5x6xf32, strided<[30, 6, 1], offset: ?>>
// CHECK: %[[SUB1:.+]] = memref.subview %[[ARG1]][%[[ARG2]], 0, 0] [1, 5, 6] [1, 1, 1] : memref<2x5x6xf32> to memref<1x5x6xf32, strided<[30, 6, 1], offset: ?>>
// CHECK: %[[COLLAPSE:.+]] = memref.collapse_shape %[[SUB]] {{\[}}[0, 1], [2]] : memref<1x5x6xf32, strided<[30, 6, 1], offset: ?>> into memref<5x6xf32, strided<[6, 1], offset: ?>>
// CHECK: %[[COLLAPSE1:.+]] = memref.collapse_shape %[[SUB1]] {{\[}}[0, 1], [2]] : memref<1x5x6xf32, strided<[30, 6, 1], offset: ?>> into memref<5x6xf32, strided<[6, 1], offset: ?>>
// CHECK: linalg.generic {indexing_maps = [#[[MAP]], #[[MAP]]], iterator_types = ["parallel", "parallel"]} ins(%[[COLLAPSE]] : memref<5x6xf32, strided<[6, 1], offset: ?>>) outs(%[[COLLAPSE1]] : memref<5x6xf32, strided<[6, 1], offset: ?>>
// CHECK: }

// CHECK: scf.for %[[ARG3:.+]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK: %[[SUB2:.+]] = memref.subview %[[ARG0]][%[[ARG3]], 0, 0] [1, 5, 6] [1, 1, 1] : memref<2x5x6xf32> to memref<1x5x6xf32, strided<[30, 6, 1], offset: ?>>
// CHECK: %[[SUB3:.+]] = memref.subview %[[ARG1]][%[[ARG3]], 0, 0] [1, 5, 6] [1, 1, 1] : memref<2x5x6xf32> to memref<1x5x6xf32, strided<[30, 6, 1], offset: ?>>
// CHECK: %[[COLLAPSE2:.+]] = memref.collapse_shape %[[SUB2]] {{\[}}[0, 1], [2]] : memref<1x5x6xf32, strided<[30, 6, 1], offset: ?>> into memref<5x6xf32, strided<[6, 1], offset: ?>>
// CHECK: %[[COLLAPSE3:.+]] = memref.collapse_shape %[[SUB3]] {{\[}}[0, 1], [2]] : memref<1x5x6xf32, strided<[30, 6, 1], offset: ?>> into memref<5x6xf32, strided<[6, 1], offset: ?>>
// CHECK: linalg.generic {indexing_maps = [#[[MAP]], #[[MAP]]], iterator_types = ["parallel", "parallel"]} ins(%[[COLLAPSE2]] : memref<5x6xf32, strided<[6, 1], offset: ?>>) outs(%[[COLLAPSE3]] : memref<5x6xf32, strided<[6, 1], offset: ?>>
// CHECK: }
