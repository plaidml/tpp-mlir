// RUN: tpp-opt %s -transform-dialect-interpreter -canonicalize | FileCheck %s

transform.sequence failures(propagate) {
  ^bb0(%arg0: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!pdl.operation) -> !pdl.operation
    %1 = transform.structured.collapse_to_2d in %0
}

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @additions(%arg0: memref<2x5x6xf32>, %arg1: memref<2x5x6xf32>) -> memref<2x5x6xf32> {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  scf.for %arg2 = %c0 to %c2 step %c1 {
     %subview = memref.subview %arg0[%arg2, 0, 0] [1, 5, 6] [1, 1, 1] : memref<2x5x6xf32> to memref<1x5x6xf32, strided<[30, 6, 1], offset: ?>>
    %subview_0 = memref.subview %arg1[%arg2, 0, 0] [1, 5, 6] [1, 1, 1] : memref<2x5x6xf32> to memref<1x5x6xf32, strided<[30, 6, 1], offset: ?>>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%subview : memref<1x5x6xf32, strided<[30, 6, 1], offset: ?>>) outs(%subview_0 : memref<1x5x6xf32, strided<[30, 6, 1], offset: ?>>) {
      ^bb0(%in: f32, %out: f32):
        %0 = arith.addf %in, %out : f32
        linalg.yield %0 : f32
    }
  }
  scf.for %arg2 = %c0 to %c2 step %c1 {
    %subview = memref.subview %arg0[%arg2, 0, 0] [1, 5, 6] [1, 1, 1] : memref<2x5x6xf32> to memref<1x5x6xf32, strided<[30, 6, 1], offset: ?>>
    %subview_0 = memref.subview %arg1[%arg2, 0, 0] [1, 5, 6] [1, 1, 1] : memref<2x5x6xf32> to memref<1x5x6xf32, strided<[30, 6, 1], offset: ?>>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%subview : memref<1x5x6xf32, strided<[30, 6, 1], offset: ?>>) outs(%subview_0 : memref<1x5x6xf32, strided<[30, 6, 1], offset: ?>>) {
      ^bb0(%in: f32, %out: f32):
        %0 = arith.addf %in, %out : f32
        linalg.yield %0 : f32
    }
  }
  return %arg1 : memref<2x5x6xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: func.func @additions(
// CHECK:  %[[ARG0:.+]]: memref<2x5x6xf32>, %[[ARG1:.+]]: memref<2x5x6xf32>) -> memref<2x5x6xf32> {
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
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
