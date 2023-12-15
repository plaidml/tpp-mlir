// RUN: tpp-opt -pack-vnni %s | FileCheck %s

func.func @brgemm(%arg0: tensor<32x4x4xbf16>, %arg1: tensor<32x4x4xbf16>, 
                  %arg2: tensor<4x4xbf16>) -> tensor<4x4xbf16>{
  %0 = linalg.batch_reduce_matmul ins(%arg0, %arg1:tensor<32x4x4xbf16>, tensor<32x4x4xbf16>) 
                                  outs(%arg2:tensor<4x4xbf16>) -> tensor<4x4xbf16>
  return %0: tensor<4x4xbf16>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3 floordiv 2, d2, d4)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d2)>

// CHECK-LABEL: brgemm
// CHECK-SAME:  %[[ARG0:.+]]: tensor<32x4x4xbf16>, %[[ARG1:.+]]: tensor<32x4x4xbf16>,
// CHECK-SAME:  %[[ARG2:.+]]: tensor<4x4xbf16>
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<32x2x4x2xbf16>
// CHECK: %[[PACK:.+]] = tensor.pack %[[ARG1]] inner_dims_pos = [1] inner_tiles = [2] 
// CHECK-SAME:  into %[[EMPTY]] : tensor<32x4x4xbf16> -> tensor<32x2x4x2xbf16>
// CHECK: %{{.+}} = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:  iterator_types = ["reduction", "parallel", "parallel", "reduction", "reduction"]
// CHECK-SAME:  ins(%[[ARG0]], %[[PACK]]
// CHECK-SAME:  outs(%[[ARG2]]
