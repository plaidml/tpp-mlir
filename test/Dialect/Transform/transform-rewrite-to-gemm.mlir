// RUN: tpp-opt %s -transform-dialect-interpreter -canonicalize | FileCheck %s

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d2, d4 + d3, d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2, d3, d6, d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>

func.func @conv(%arg0: tensor<1x4x4x3xi64>, %arg1: tensor<2x2x3x8xi64>, %arg2: tensor<1x3x3x8xi64>) -> tensor<1x3x3x8xi64> {
  // CHECK: linalg.matmul
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<1x4x4x3xi64>, tensor<2x2x3x8xi64>) outs(%arg2 : tensor<1x3x3x8xi64>) {
  ^bb0(%in: i64, %in_0: i64, %out: i64):
    %1 = arith.muli %in, %in_0 : i64
    %2 = arith.addi %out, %1 : i64
    linalg.yield %2 : i64
  } -> tensor<1x3x3x8xi64>
  return %0 : tensor<1x3x3x8xi64>
}

transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.rewrite_conv_to_matmul %0 : !transform.any_op
}
