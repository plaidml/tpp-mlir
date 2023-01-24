// RUN: tpp-opt %s -interchange-conv-to-expose-matmul -split-input-file | FileCheck %s

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>

func.func @blck_conv(%arg0: tensor<1x2x56x56x32xi64>, %arg1: tensor<2x2x1x1x32x32xi64>, 
                     %arg2: tensor<1x2x56x56x32xi64>) -> tensor<1x2x56x56x32xi64> {
  %0 = linalg.generic {
    indexing_maps = [#map, #map1, #map2], 
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} 
    ins(%arg0, %arg1 : tensor<1x2x56x56x32xi64>, tensor<2x2x1x1x32x32xi64>) outs(%arg2 : tensor<1x2x56x56x32xi64>) {
    ^bb0(%in: i64, %in_51: i64, %out: i64):
      %151 = arith.muli %in, %in_51 : i64
      %152 = arith.addi %out, %151 : i64
      linalg.yield %152 : i64
  } -> tensor<1x2x56x56x32xi64>
  return %0 : tensor<1x2x56x56x32xi64>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d3, d2 + d4, d6 + d5, d8)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d3, d4, d5, d8, d7)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d6, d7)>
// CHECK: func.func @blck_conv(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<1x2x56x56x32xi64>, %[[ARG1:.+]]: tensor<2x2x1x1x32x32xi64>,
// CHECK-SAME:  %[[ARG2:.+]]: tensor<1x2x56x56x32xi64>)
// CHECK: %{{.+}} = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "reduction", 
// CHECK-SAME:                    "reduction", "reduction", "parallel", "parallel", "reduction"]
// CHECK-SAME:  ins(%[[ARG0]], %[[ARG1]]
// CHECK-SAME:  outs(%[[ARG2]]

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>

func.func @blck_conv(%arg0: tensor<1x2x56x56x32xi64>, %arg1: tensor<2x2x1x1x32x32xi64>,
                     %arg2: tensor<1x2x56x56x32xi64>) -> tensor<1x2x56x56x32xi64> {
  %0 = linalg.generic {
    indexing_maps = [#map, #map1, #map2], 
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} 
    ins(%arg0, %arg1 : tensor<1x2x56x56x32xi64>, tensor<2x2x1x1x32x32xi64>) outs(%arg2 : tensor<1x2x56x56x32xi64>) {
    ^bb0(%in: i64, %in_51: i64, %out: i64):
      %151 = arith.muli %in, %in_51 : i64
      %152 = arith.subi %out, %151 : i64
      linalg.yield %152 : i64
  } -> tensor<1x2x56x56x32xi64>
  return %0 : tensor<1x2x56x56x32xi64>
}

// Not a conv.
// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
// CHECK: func.func @blck_conv(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<1x2x56x56x32xi64>, %[[ARG1:.+]]: tensor<2x2x1x1x32x32xi64>,
// CHECK-SAME:  %[[ARG2:.+]]: tensor<1x2x56x56x32xi64>)
// CHECK: %{{.+}} = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel", 
// CHECK-SAME:                    "parallel", "reduction", "reduction", "reduction", "reduction"]
// CHECK-SAME:  ins(%[[ARG0]], %[[ARG1]]
// CHECK-SAME:  outs(%[[ARG2]]

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d1, d2, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d5, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>

func.func @blck_conv(%arg0: tensor<1x2x56x56x32xi64>, %arg1: tensor<2x2x1x1x32x32xi64>, %arg2: tensor<1x2x56x56x32xi64>) -> tensor<1x2x56x56x32xi64> {
  %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2], [3], [4]] : tensor<1x2x56x56x32xi64> into tensor<2x56x56x32xi64>
  %collapsed_0 = tensor.collapse_shape %arg1 [[0], [1, 2, 3], [4], [5]] : tensor<2x2x1x1x32x32xi64> into tensor<2x2x32x32xi64>
  %collapsed_1 = tensor.collapse_shape %arg2 [[0, 1], [2], [3], [4]] : tensor<1x2x56x56x32xi64> into tensor<2x56x56x32xi64>
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%collapsed, %collapsed_0 : tensor<2x56x56x32xi64>, tensor<2x2x32x32xi64>) outs(%collapsed_1 : tensor<2x56x56x32xi64>) {
    ^bb0(%in: i64, %in_2: i64, %out: i64):
      %1 = arith.muli %in, %in_2 : i64
      %2 = arith.subi %out, %1 : i64
      linalg.yield %2 : i64
  } -> tensor<2x56x56x32xi64>
  %expanded = tensor.expand_shape %0 [[0, 1], [2], [3], [4]] : tensor<2x56x56x32xi64> into tensor<1x2x56x56x32xi64>
  return %expanded : tensor<1x2x56x56x32xi64>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d1, d2, d5)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d5, d3)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
// CHECK: func.func @blck_conv(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<1x2x56x56x32xi64>,
// CHECK-SAME:  %[[ARG1:.+]]: tensor<2x2x1x1x32x32xi64>,
// CHECK-SAME:  %[[ARG2:.+]]: tensor<1x2x56x56x32xi64>)
// CHECK: %[[CLPS:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1], [2], [3], [4]] 
// CHECK-SAME:  : tensor<1x2x56x56x32xi64> into tensor<2x56x56x32xi64>
// CHECK: %[[CLPS0:.+]] = tensor.collapse_shape %[[ARG1]] {{\[}}[0], [1, 2, 3], [4], [5]] 
// CHECK-SAME:  : tensor<2x2x1x1x32x32xi64> into tensor<2x2x32x32xi64>
// CHECK: %[[CLPS1:.+]] = tensor.collapse_shape %[[ARG2]] {{\[}}[0, 1], [2], [3], [4]] 
// CHECK-SAME:  : tensor<1x2x56x56x32xi64> into tensor<2x56x56x32xi64>
// CHECK: %{{.+}} = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]
// CHECK-SAME:  ins(%[[CLPS]], %[[CLPS0]]
// CHECK-SAME:  outs(%[[CLPS1]]
