// RUN: tpp-opt %s -transform-dialect-interpreter -canonicalize -split-input-file | FileCheck %s

#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

transform.sequence failures(propagate) {
  ^bb0(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.rewrite_to_brgemm %0 : !transform.any_op
}

// CHECK-LABEL: func.func @blocked_matmul(
// CHECK-SAME: %[[ARG0:.+]]: tensor<4x16x32x32xf32>,
// CHECK-SAME: %[[ARG1:.+]]: tensor<8x16x32x32xf32>,
// CHECK-SAME: %[[ARG2:.+]]: tensor<4x8x32x32xf32>)
func.func @blocked_matmul(%arg0: tensor<4x16x32x32xf32>, %arg1: tensor<8x16x32x32xf32>, %arg2: tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> {
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[C8:.+]] = arith.constant 8 : index
  // CHECK: %[[OUTER:.+]] = scf.for %[[P1:.+]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%[[INIT:.+]] = %[[ARG2]]) -> (tensor<4x8x32x32xf32>) {
  // CHECK: %[[INNER:.+]] = scf.for %[[P2:.+]] = %[[C0]] to %[[C8]] step %[[C1]] iter_args(%[[INIT2:.+]] = %[[INIT]]) -> (tensor<4x8x32x32xf32>) {
  // CHECK: %[[SLICEA:.+]] = tensor.extract_slice %[[ARG0]][%[[P1]], 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : tensor<4x16x32x32xf32> to tensor<16x32x32xf32>
  // CHECK: %[[SLICEB:.+]] = tensor.extract_slice %[[ARG1]][%[[P2]], 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : tensor<8x16x32x32xf32> to tensor<16x32x32xf32>
  // CHECK: %[[SLICEC:.+]] = tensor.extract_slice %[[INIT2]][%[[P1]], %[[P2]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<4x8x32x32xf32> to tensor<32x32xf32>
  // CHECK: %[[MUL:.+]] = linalg.batch_reduce_matmul ins(%[[SLICEA]], %[[SLICEB]] : tensor<16x32x32xf32>, tensor<16x32x32xf32>) outs(%[[SLICEC]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[YIELD:.+]] = tensor.insert_slice %[[MUL]] into %[[INIT2]][%[[P1]], %[[P2]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xf32> into tensor<4x8x32x32xf32>
  // CHECK: scf.yield %[[YIELD]] : tensor<4x8x32x32xf32>
  // CHECK: }
  // CHECK: scf.yield %[[INNER]] : tensor<4x8x32x32xf32>
  // CHECK: }
  // CHECK: return %[[OUTER]] : tensor<4x8x32x32xf32>
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

transform.sequence failures(propagate) {
  ^bb0(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.rewrite_to_brgemm %0 : !transform.any_op
}

// CHECK-LABEL: func.func @blocked_matmul(
// CHECK-SAME:  %[[ARG0:.+]]: memref<4x16x32x32xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<8x16x32x32xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: memref<4x8x32x32xf32>) -> memref<4x8x32x32xf32> {
func.func @blocked_matmul(%arg0: memref<4x16x32x32xf32>, %arg1: memref<8x16x32x32xf32>, %arg2: memref<4x8x32x32xf32>) -> memref<4x8x32x32xf32> {
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[C8:.+]] = arith.constant 8 : index
  // CHECK: scf.parallel (%[[P1:.+]], %[[P2:.+]]) = (%[[C0]], %[[C0]]) to (%[[C4]], %[[C8]]) step (%[[C1]], %[[C1]]) {
  // CHECK-NEXT: %[[SLICE1:.+]] = memref.subview %[[ARG0]][%[[P1]], 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<4x16x32x32xf32> to memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>
  // CHECK-NEXT: %[[SLICE2:.+]] = memref.subview %[[ARG1]][%[[P2]], 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<8x16x32x32xf32> to memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>
  // CHECK-NEXT: %[[SLICE3:.+]] = memref.subview %[[ARG2]][%[[P1]], %[[P2]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<4x8x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
  // CHECK-NEXT: linalg.batch_reduce_matmul ins(%[[SLICE1]], %[[SLICE2]] : memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>) outs(%[[SLICE3]] : memref<32x32xf32, strided<[32, 1], offset: ?>>)
  // CHECK: }
  // CHECK-NEXT: return %[[ARG2]] : memref<4x8x32x32xf32>
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

transform.sequence failures(propagate) {
  ^bb0(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.rewrite_to_brgemm %0 : !transform.any_op
}

// CHECK-LABEL: func.func @blocked_matmul(
// CHECK-SAME: %[[ARG0:.+]]: tensor<8x32x32xf32>, %[[ARG1:.+]]: tensor<8x32x32xf32>, %[[ARG2:.+]]: tensor<32x32xf32>) -> tensor<32x32xf32> {
func.func @blocked_matmul(%arg0: tensor<8x32x32xf32>, %arg1: tensor<8x32x32xf32>, %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK: %[[MUL:.+]] = linalg.batch_reduce_matmul ins(%[[ARG0]], %[[ARG1]] : tensor<8x32x32xf32>, tensor<8x32x32xf32>) outs(%[[ARG2]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  %0 = linalg.generic {indexing_maps = [#map5, #map6, #map7], iterator_types = ["reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<8x32x32xf32>, tensor<8x32x32xf32>) outs(%arg2 : tensor<32x32xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5:f32):
    %m = arith.mulf %arg3, %arg4 : f32
    %a = arith.addf %arg5, %m : f32
    linalg.yield %a : f32
  } -> tensor<32x32xf32>
  // CHECK: return %[[MUL]] : tensor<32x32xf32>
  return %0: tensor<32x32xf32>
}

// -----

#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4)>

transform.sequence failures(propagate) {
  ^bb0(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.rewrite_to_brgemm %0 : !transform.any_op
}

// CHECK-LABEL: func.func @blocked_matmul(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<4x8x32x32xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: tensor<16x8x32x32xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: tensor<32x32xf32>)
func.func @blocked_matmul(%arg0: tensor<4x8x32x32xf32>, %arg1: tensor<16x8x32x32xf32>, %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[C16:.+]] = arith.constant 16 : index
  // CHECK: %[[OUTER:.+]] = scf.for %[[ARG3:.+]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%[[ARG4:.+]] = %[[ARG2]]) -> (tensor<32x32xf32>) {
  // CHECK: %[[INNER:.+]] = scf.for %[[ARG5:.+]] = %[[C0]] to %[[C16]] step %[[C1]] iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<32x32xf32>) {
  // CHECK: %[[SLICEA:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG3]], 0, 0, 0] [1, 8, 32, 32] [1, 1, 1, 1] : tensor<4x8x32x32xf32> to tensor<8x32x32xf32>
  // CHECK: %[[SLICEB:.+]] = tensor.extract_slice %[[ARG1]][%[[ARG5]], 0, 0, 0] [1, 8, 32, 32] [1, 1, 1, 1] : tensor<16x8x32x32xf32> to tensor<8x32x32xf32>
  // CHECK: %[[MUL:.+]] = linalg.batch_reduce_matmul ins(%[[SLICEA]], %[[SLICEB]] : tensor<8x32x32xf32>, tensor<8x32x32xf32>) outs(%[[ARG6]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: scf.yield %[[MUL]] : tensor<32x32xf32>
  // CHECK: }
  // CHECK: scf.yield %[[INNER]] : tensor<32x32xf32>
  // CHECK: return %[[OUTER]] : tensor<32x32xf32>
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

transform.sequence failures(propagate) {
  ^bb0(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.rewrite_to_brgemm %0 : !transform.any_op
}

// CHECK-LABEL: func.func @blocked_matmul(
// CHECK-SAME: %[[ARG0:.+]]: tensor<4x8x32x32xf32>,
// CHECK-SAME: %[[ARG1:.+]]: tensor<16x8x32x32xf32>,
// CHECK-SAME: %[[ARG2:.+]]: tensor<16x32x32xf32>)
func.func @blocked_matmul(%arg0: tensor<4x8x32x32xf32>, %arg1: tensor<16x8x32x32xf32>, %arg2: tensor<16x32x32xf32>) -> tensor<16x32x32xf32> {
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[C16:.+]] = arith.constant 16 : index
  // CHECK: %[[OUTER:.+]] = scf.for %[[ARG3:.+]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%[[ARG4:.+]] = %[[ARG2]]) -> (tensor<16x32x32xf32>) {
  // CHECK: %[[INNER:.+]] = scf.for %[[ARG5:.+]] = %[[C0]] to %[[C16]] step %[[C1]] iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<16x32x32xf32>) {
  // CHECK: %[[SLICEA:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG3]], 0, 0, 0] [1, 8, 32, 32] [1, 1, 1, 1] : tensor<4x8x32x32xf32> to tensor<8x32x32xf32>
  // CHECK: %[[SLICEB:.+]] = tensor.extract_slice %[[ARG1]][%[[ARG5]], 0, 0, 0] [1, 8, 32, 32] [1, 1, 1, 1] : tensor<16x8x32x32xf32> to tensor<8x32x32xf32>
  // CHECK: %[[SLICEC:.+]] = tensor.extract_slice %[[ARG6]][%[[ARG5]], 0, 0] [1, 32, 32] [1, 1, 1] : tensor<16x32x32xf32> to tensor<32x32xf32>
  // CHECK: %[[MUL:.+]] = linalg.batch_reduce_matmul ins(%[[SLICEA]], %[[SLICEB]] : tensor<8x32x32xf32>, tensor<8x32x32xf32>) outs(%[[SLICEC]] : tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[INSERT:.*]] = tensor.insert_slice %[[MUL]] into %[[ARG6]][%[[ARG5]], 0, 0] [1, 32, 32] [1, 1, 1] : tensor<32x32xf32> into tensor<16x32x32xf32>
  // CHECK: scf.yield %[[INSERT]] : tensor<16x32x32xf32>
  // CHECK: }
  // CHECK: scf.yield %[[INNER]] : tensor<16x32x32xf32>
  // CHECK: }
  // CHECK: return %[[OUTER]] : tensor<16x32x32xf32>
  %0 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1: tensor<4x8x32x32xf32>, tensor<16x8x32x32xf32>) outs(%arg2: tensor<16x32x32xf32>) {
  ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
    %mul = arith.mulf %arg4, %arg5 : f32
    %add = arith.addf %arg6, %mul : f32
    linalg.yield %add : f32
  } -> tensor<16x32x32xf32>
  return %0 : tensor<16x32x32xf32>
}

// -----

#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>

transform.sequence failures(propagate) {
  ^bb0(%arg1: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.rewrite_to_brgemm %0 : !transform.any_op
}

func.func @blocked_matmul(%arg0: tensor<?x32x32xf32>, %arg1: tensor<?x32x32xf32>, %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = linalg.generic {indexing_maps = [#map5, #map6, #map7], iterator_types = ["reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<?x32x32xf32>, tensor<?x32x32xf32>) outs(%arg2 : tensor<32x32xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5:f32):
    %m = arith.mulf %arg3, %arg4 : f32
    %a = arith.addf %arg5, %m : f32
    linalg.yield %a : f32
  } -> tensor<32x32xf32>
  return %0: tensor<32x32xf32>
}

// CHECK: func.func @blocked_matmul(
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x32x32xf32>, %[[ARG1:.+]]: tensor<?x32x32xf32>, %[[ARG2:.+]]: tensor<32x32xf32>)
// CHECK: %[[MUL:.+]] = linalg.batch_reduce_matmul ins(%[[ARG0]], %[[ARG1]] : tensor<?x32x32xf32>, tensor<?x32x32xf32>) outs(%[[ARG2]] : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK: return %[[MUL]] : tensor<32x32xf32>
