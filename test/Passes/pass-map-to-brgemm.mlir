// RUN: tpp-opt %s -rewrite-to-brgemm -split-input-file | FileCheck %s

#map0 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

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
  %1 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<4x16x32x32xf32>, tensor<8x16x32x32xf32>) outs(%arg2 : tensor<4x8x32x32xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %8 = arith.mulf %arg3, %arg4 : f32
      %9 = arith.addf %arg5, %8 : f32
      linalg.yield %9 : f32
    } -> tensor<4x8x32x32xf32>
  return %1 :  tensor<4x8x32x32xf32>
}

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4 floordiv 2, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>

// CHECK-LABEL: func.func @vnni_layout_brgemm
// CHECK-SAME:  %[[ARG0:.+]]: tensor<48x32x32xbf16>,
// CHECK-SAME:  %[[ARG1:.+]]: tensor<48x16x32x2xbf16>,
// CHECK-SAME:  %[[ARG2:.+]]: tensor<32x32xbf16>
func.func @vnni_layout_brgemm(%arg0: tensor<48x32x32xbf16>, 
                              %arg1: tensor<48x16x32x2xbf16>, %arg2: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
  %0 = linalg.generic {
    indexing_maps = [#map, #map1, #map2], 
    iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]} 
    ins(%arg0, %arg1 : tensor<48x32x32xbf16>, tensor<48x16x32x2xbf16>) 
    outs(%arg2 : tensor<32x32xbf16>) {
      ^bb0(%in: bf16, %in_8: bf16, %out: bf16):
        %11 = arith.mulf %in, %in_8 : bf16
        %12 = arith.addf %out, %11 : bf16
        linalg.yield %12 : bf16
  } -> tensor<32x32xbf16>
  // CHECK: vnni.brgemm ins(%[[ARG0]] : tensor<48x32x32xbf16>, %[[ARG1]] : tensor<48x16x32x2xbf16>) 
  // CHECK-SAME:        out(%[[ARG2]] : tensor<32x32xbf16>) -> tensor<32x32xbf16>
  return %0 : tensor<32x32xbf16>
}
