// RUN: tpp-opt %s -split-input-file -tile-consumer-and-fuse-producers -cse | FileCheck %s

// CHECK-NOT: scf.for
func.func @matmul_sequence_fusion(%arg0: tensor<32x64xf32>, %arg1: tensor<64x32xf32>,
    %arg2: tensor<32x32xf32>, %arg3: tensor<32x64xf32>, %arg4: tensor<32x64xf32>,
    %arg5: tensor<64x32xf32>, %arg6: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<32x64xf32>, tensor<64x32xf32>)
    outs(%arg2 : tensor<32x32xf32>) -> tensor<32x32xf32> // [M, N0] * [N0, N1]
  %1 = linalg.matmul ins(%0, %arg3 : tensor<32x32xf32>, tensor<32x64xf32>)
    outs(%arg4 : tensor<32x64xf32>) -> tensor<32x64xf32> // [M, N1] * [N1, N2]
  %2 = linalg.matmul ins(%1, %arg5 : tensor<32x64xf32>, tensor<64x32xf32>)
    outs(%arg6 : tensor<32x32xf32>) -> tensor<32x32xf32> // [M, N2] * [N2, N3]
  return %2 : tensor<32x32xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @matmul_eletwise(%arg0: tensor<32x64xf32>, %arg1: tensor<64x32xf32>,
    %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<32x64xf32>, tensor<64x32xf32>)
    outs(%arg2 : tensor<32x32xf32>) -> tensor<32x32xf32>
  %1 = linalg.generic {indexing_maps = [#map], 
                       iterator_types = ["parallel", "parallel"]} 
    outs(%0: tensor<32x32xf32>) {
      ^bb0(%out: f32):
        %2 = arith.maxf %out, %c0 : f32
        linalg.yield %2 : f32
    } -> tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<() -> ()>
// CHECK: func.func @matmul_eletwise
// CHECK-SAME:  %[[ARG0:.+]]: tensor<32x64xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: tensor<64x32xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: tensor<32x32xf32>)
// CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[LOOP:.+]] = scf.for %[[ARG3:.+]] = %[[C0]] to %[[C32]] step %[[C1]] iter_args(%[[ARG4:.+]] = %[[ARG2]]) -> (tensor<32x32xf32>) {
// CHECK: %[[LOOP0:.+]] = scf.for %[[ARG5:.+]] = %[[C0]] to %[[C32]] step %[[C1]] iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<32x32xf32>) {
// CHECK: %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG3]], 0] [1, 64] [1, 1] : tensor<32x64xf32> to tensor<1x64xf32>
// CHECK: %[[SLICE0:.+]] = tensor.extract_slice %[[ARG1]][0, %[[ARG5]]] [64, 1] [1, 1] : tensor<64x32xf32> to tensor<64x1xf32>
// CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[ARG6]][%[[ARG3]], %[[ARG5]]] [1, 1] [1, 1] : tensor<32x32xf32> to tensor<1x1xf32>
// CHECK: %[[MUL:.+]] = linalg.matmul ins(%[[SLICE]], %[[SLICE0]] : tensor<1x64xf32>, tensor<64x1xf32>) outs(%[[SLICE1]] : tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK: %[[SLICE2:.+]] = tensor.extract_slice %[[MUL]][0, 0] [1, 1] [1, 1] : tensor<1x1xf32> to tensor<f32>
// CHECK: linalg.generic {indexing_maps = [#[[MAP]]], iterator_types = []} outs(%[[SLICE2]] : tensor<f32>)

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @matmul_eletwise(%arg0: tensor<4x4x32x32xf32>, %arg1: tensor<4x4x32x32xf32>,
    %arg2: tensor<4x4x32x32xf32>) -> tensor<4x4x32x32xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.generic {
      indexing_maps = [#map, #map1, #map2], 
      iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} 
      ins(%arg0, %arg1 : tensor<4x4x32x32xf32>, tensor<4x4x32x32xf32>) 
      outs(%arg2 : tensor<4x4x32x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %1 = arith.mulf %in, %in_2 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
  } -> tensor<4x4x32x32xf32>
  %3 = linalg.generic {
      indexing_maps = [#map3],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      outs(%0 : tensor<4x4x32x32xf32>) {
    ^bb0(%out: f32):
      %4 = arith.maxf %out, %c0 : f32
      linalg.yield %4 : f32
  } -> tensor<4x4x32x32xf32>
  return %3 : tensor<4x4x32x32xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// CHECK: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
// CHECK: #[[MAP3:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: func.func @matmul_eletwise(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<4x4x32x32xf32>, %[[ARG1:.+]]: tensor<4x4x32x32xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: tensor<4x4x32x32xf32>)
// CHECK: %[[C4:.+]] = arith.constant 4 : index
// CHECK: %[[C0:.+]] = arith.constant 0 : index
// CHECK: %[[C1:.+]] = arith.constant 1 : index
// CHECK: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[LOOP:.+]] = scf.for %[[ARG3:.+]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%[[ARG4:.+]] = %[[ARG2]]) -> (tensor<4x4x32x32xf32>) {
// CHECK: %[[LOOP1:.+]] = scf.for %[[ARG5:.+]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<4x4x32x32xf32>) {
// CHECK: %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG3]], 0, 0, 0] [1, 4, 32, 32] [1, 1, 1, 1] : tensor<4x4x32x32xf32> to tensor<4x32x32xf32>
// CHECK: %[[SLICE0:.+]] = tensor.extract_slice %[[ARG1]][%[[ARG5]], 0, 0, 0] [1, 4, 32, 32] [1, 1, 1, 1] : tensor<4x4x32x32xf32> to tensor<4x32x32xf32> 
// CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[ARG6]][%[[ARG3]], %[[ARG5]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<4x4x32x32xf32> to tensor<32x32xf32>
// CHECK: %[[MUL:.+]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]], iterator_types = ["reduction", "parallel", "parallel", "reduction"]} ins(%[[SLICE]], %[[SLICE0]] : tensor<4x32x32xf32>, tensor<4x32x32xf32>) outs(%[[SLICE1]] : tensor<32x32xf32>)
// CHECK: %[[ELEM:.+]] = linalg.generic {indexing_maps = [#[[MAP3]]], iterator_types = ["parallel", "parallel"]} outs(%[[MUL]] : tensor<32x32xf32>) 
