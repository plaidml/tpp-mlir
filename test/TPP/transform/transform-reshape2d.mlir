// RUN: tpp-opt %s -transform-dialect-interpreter -canonicalize -split-input-file -verify-diagnostics | FileCheck %s

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
    %1 = transform.structured.reshape_2d in %0
}

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @add(%arg0: memref<5x5x4xf32>, %arg1: memref<5x5x4xf32>) -> memref<5x5x4xf32> {
  linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%arg0: memref<5x5x4xf32>) outs(%arg1: memref<5x5x4xf32>) {
      ^bb0(%arg2: f32, %arg3: f32):
        %1 = arith.addf %arg2, %arg3 : f32
        linalg.yield %1 : f32
  }
  return %arg1: memref<5x5x4xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: func.func @add(
// CHECK-SAME: %[[ARG0:.+]]: memref<5x5x4xf32>, %[[ARG1:.+]]: memref<5x5x4xf32>) -> memref<5x5x4xf32> {
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C5:.+]] = arith.constant 5 : index
// CHECK: scf.for %[[ARG2:.+]] = %[[C0]] to %[[C5]] step %[[C1]] {
// CHECK: %[[SUB:.+]] = memref.subview %[[ARG0]][%[[ARG2]], 0, 0] [1, 5, 4] [1, 1, 1] : memref<5x5x4xf32> to memref<1x5x4xf32, strided<[20, 4, 1], offset: ?>>
// CHECK: %[[SUB0:.+]] = memref.subview %[[ARG1]][%[[ARG2]], 0, 0] [1, 5, 4] [1, 1, 1] : memref<5x5x4xf32> to memref<1x5x4xf32, strided<[20, 4, 1], offset: ?>>
// CHECK: linalg.generic {indexing_maps = [#[[MAP]], #[[MAP]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[SUB]] : memref<1x5x4xf32, strided<[20, 4, 1], offset: ?>>) outs(%[[SUB0]] : memref<1x5x4xf32, strided<[20, 4, 1], offset: ?>>)
// CHECK: return %[[ARG1]] : memref<5x5x4xf32>
// CHECK: }

// -----

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
    %1 = transform.structured.reshape_2d in %0
}

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @add(%arg0: tensor<5x5x4xf32>, %arg1: tensor<5x5x4xf32>) -> tensor<5x5x4xf32> {
  %0 = linalg.generic { indexing_maps = [#map, #map], 
                        iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%arg0: tensor<5x5x4xf32>) outs(%arg1: tensor<5x5x4xf32>) {
      ^bb0(%arg2: f32, %arg3: f32):
        %1 = arith.addf %arg2, %arg3 : f32
        linalg.yield %1 : f32
  } -> tensor<5x5x4xf32>     
  return %0: tensor<5x5x4xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: func.func @add(
// CHECK-SAME: %[[ARG0:.+]]: tensor<5x5x4xf32>, %[[ARG1:.+]]: tensor<5x5x4xf32>) -> tensor<5x5x4xf32> {
// CHECK: %[[LOOP:.+]] = scf.for %[[ARG2:.+]] = %[[C0]] to %[[C5]] step %[[C1]] iter_args(%[[ARG3:.+]] = %[[ARG1]]) -> (tensor<5x5x4xf32>) {
// CHECK: %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG2]], 0, 0] [1, 5, 4] [1, 1, 1] : tensor<5x5x4xf32> to tensor<1x5x4xf32>
// CHECK: %[[SLICE0:.+]] = tensor.extract_slice %[[ARG3]][%[[ARG2]], 0, 0] [1, 5, 4] [1, 1, 1] : tensor<5x5x4xf32> to tensor<1x5x4xf32>
// CHECK: %[[ADD:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[SLICE]] : tensor<1x5x4xf32>) outs(%[[SLICE0]] : tensor<1x5x4xf32>)
// CHECK: %[[INSERT:.+]] = tensor.insert_slice %[[ADD]] into %[[ARG3]][%[[ARG2]], 0, 0] [1, 5, 4] [1, 1, 1] : tensor<1x5x4xf32> into tensor<5x5x4xf32>
// CHECK: scf.yield %[[INSERT]] : tensor<5x5x4xf32>
// CHECK: }
// CHECK: return %[[LOOP]] : tensor<5x5x4xf32>
// CHECK: }

// -----

// Expect to fail, as we have a single loop
transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
    // expected-error @below {{Expect at least two loops:}}
    %1 = transform.structured.reshape_2d in %0
}

#map = affine_map<(d0) -> (d0)>
func.func @add(%arg0: tensor<5xf32>, %arg1: tensor<5xf32>) -> tensor<5xf32> {
  // expected-note @below {{when applied to this op}}
  %0 = linalg.generic { indexing_maps = [#map, #map], 
                        iterator_types = ["parallel"]}
    ins(%arg0: tensor<5xf32>) outs(%arg1: tensor<5xf32>) {
      ^bb0(%arg2: f32, %arg3: f32):
        %1 = arith.addf %arg2, %arg3 : f32
        linalg.yield %1 : f32
  } -> tensor<5xf32>     
  return %0: tensor<5xf32>
}

// -----

// Expect to fail as we handle only generic
transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
    // expected-error @below {{Cannot reshape non-generic:}}
    %1 = transform.structured.reshape_2d in %0
}

func.func @block_linalg_matmul(
  %arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>)
    -> tensor<?x?xf32> {
  // expected-note @below {{when applied to this op}}
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%arg2: tensor<?x?xf32>)
    -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1
    %1 = transform.structured.reshape_2d in %0
}

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @add(%arg0: tensor<?x?x4xf32>, %arg1: tensor<?x?x4xf32>) -> tensor<?x?x4xf32> {
  %0 = linalg.generic { indexing_maps = [#map, #map],
                        iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%arg0: tensor<?x?x4xf32>) outs(%arg1: tensor<?x?x4xf32>) {
      ^bb0(%arg2: f32, %arg3: f32):
        %1 = arith.addf %arg2, %arg3 : f32
        linalg.yield %1 : f32
  } -> tensor<?x?x4xf32>
  return %0: tensor<?x?x4xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: func.func @add(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<?x?x4xf32>, %[[ARG1:.+]]: tensor<?x?x4xf32>) -> tensor<?x?x4xf32> {
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[DIM:.+]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?x4xf32>
// CHECK: %[[LOOP:.+]] = scf.for %[[ARG2:.+]] = %[[C0]] to %[[DIM]] step %[[C1]] iter_args(%[[ARG3:.+]] = %[[ARG1]]) -> (tensor<?x?x4xf32>) {
// CHECK: %[[DIM0:.+]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x4xf32>
// CHECK-NEXT: %[[DIM1:.+]] = tensor.dim %[[ARG3]], %[[C1]] : tensor<?x?x4xf32>
// CHECK: %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG2]], 0, 0] [1, %[[DIM0]], 4] [1, 1, 1] : tensor<?x?x4xf32> to tensor<1x?x4xf32>
// C_HECK: %[[SLICE1:.+]] = tensor.extract_slice %[[ARG3]][%[[ARG2]], 0, 0] [1, %[[DIM1]], 4] [1, 1, 1] : tensor<?x?x4xf32> to tensor<1x?x4xf32>
// C_HECK: %[[ADD:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[SLICE]] : tensor<1x?x4xf32>) outs(%[[SLICE1]] : tensor<1x?x4xf32>)
// CHECK: %[[INSERT:.+]] = tensor.insert_slice %[[ADD]] into %[[ARG3]][%[[ARG2]], 0, 0] [1, %[[DIM1]], 4] [1, 1, 1] : tensor<1x?x4xf32> into tensor<?x?x4xf32>
// CHECK: scf.yield %[[INSERT]] : tensor<?x?x4xf32>
// CHECK: }
// CHECK: return %[[LOOP]] : tensor<?x?x4xf32>
// CHECK: }
