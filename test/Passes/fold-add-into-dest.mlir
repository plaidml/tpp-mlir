// RUN: tpp-opt %s -fold-add-into-dest -cse -split-input-file | FileCheck %s

!type = tensor<2048x2048xf32>
func.func @expect_add_to_fold(%arg0: !type, %arg1: !type) -> !type {
  %0 = arith.constant dense<1.111111e+00> : !type
  %cst = arith.constant 0.000000e+00 : f32
  %1 = tensor.empty() : !type
  %2 = linalg.fill ins(%cst : f32) outs(%1 : !type) -> !type
  %3 = linalg.matmul ins(%arg0, %0 : !type, !type) outs(%2 : !type) -> !type
  %4 = linalg.matmul ins(%arg1, %0 : !type, !type) outs(%2 : !type) -> !type
  %5 = linalg.add ins(%3, %4 : !type, !type) outs(%1 : !type) -> !type
  return %5 : !type
}

// CHECK-LABEL: func.func @expect_add_to_fold
// CHECK: %[[ACC:.+]] = linalg.matmul
// CHECK-NEXT: %[[RES:.+]] = linalg.matmul ins(%[[X:.+]]) outs(%[[ACC]]
// CHECK-NEXT: return %[[RES]]

// -----

!type = tensor<2048x2048xf32>
func.func @expect_add_to_fold(%arg0: !type, %arg1: !type) -> !type {
  %0 = arith.constant dense<1.111111e+00> : !type
  %cst = arith.constant 0.000000e+00 : f32
  %1 = tensor.empty() : !type
  %2 = linalg.fill ins(%cst : f32) outs(%1 : !type) -> !type
  %3 = linalg.matmul ins(%arg0, %0 : !type, !type) outs(%2 : !type) -> !type
  %5 = linalg.add ins(%3, %arg1 : !type, !type) outs(%1 : !type) -> !type
  return %5 : !type
}

// CHECK-LABEL: func.func @expect_add_to_fold
// CHECK: %[[RES:.+]] = linalg.matmul
// CHECK-NEXT: return %[[RES]]

// -----

!type = tensor<2048x2048xf32>
func.func @expect_add_to_fold(%arg0: !type, %arg1: !type) -> !type {
  %0 = arith.constant dense<1.111111e+00> : !type
  %cst = arith.constant 0.000000e+00 : f32
  %1 = tensor.empty() : !type
  %2 = linalg.fill ins(%cst : f32) outs(%1 : !type) -> !type
  %3 = linalg.matmul_transpose_a ins(%arg0, %0 : !type, !type) outs(%2 : !type) -> !type
  %4 = linalg.matmul_transpose_b ins(%arg1, %0 : !type, !type) outs(%2 : !type) -> !type
  %5 = linalg.add ins(%3, %4 : !type, !type) outs(%1 : !type) -> !type
  return %5 : !type
}

// CHECK-LABEL: func.func @expect_add_to_fold
// CHECK: %[[ACC:.+]] = linalg.matmul_transpose_a
// CHECK-NEXT: %[[RES:.+]] = linalg.matmul_transpose_b ins(%[[X:.+]]) outs(%[[ACC]]
// CHECK-NEXT: return %[[RES]]

// -----

!type = tensor<2048x2048xf32>
func.func @expect_no_fold_as_operands_do_not_dominate_each_other(%arg0: !type, %arg1: !type) -> !type {
  %0 = arith.constant dense<1.111111e+00> : !type
  %cst = arith.constant 0.000000e+00 : f32
  %1 = tensor.empty() : !type
  %2 = linalg.fill ins(%cst : f32) outs(%1 : !type) -> !type
  %3 = linalg.matmul_transpose_b ins(%arg0, %0 : !type, !type) outs(%2 : !type) -> !type
  %4 = linalg.add ins(%3, %3 : !type, !type) outs(%1 : !type) -> !type
  return %4 : !type
}


// CHECK-LABEL: func.func @expect_no_fold_as_operands_do_not_dominate_each_other
// CHECK: linalg.fill
// CHECK-NEXT: linalg.matmul_transpose_b
// CHECK-NEXT: linalg.add
// CHECK-NEXT: return

// -----

!type = tensor<2048x2048xf32>
func.func @expect_no_fold_as_dominated_op_is_not_a_contraction(%arg0: !type, %arg1: !type) -> !type {
  %0 = arith.constant dense<1.111111e+00> : !type
  %cst = arith.constant 0.000000e+00 : f32
  %1 = tensor.empty() : !type
  %2 = linalg.fill ins(%cst : f32) outs(%1 : !type) -> !type
  %3 = linalg.matmul ins(%arg0, %0 : !type, !type) outs(%2 : !type) -> !type
  %4 = linalg.sub ins(%arg1, %0 : !type, !type) outs(%2 : !type) -> !type
  %5 = linalg.add ins(%3, %4 : !type, !type) outs(%1 : !type) -> !type
  return %5 : !type
}

// CHECK-LABEL: func.func @expect_no_fold_as_dominated_op_is_not_a_contraction
// CHECK: linalg.fill
// CHECK-NEXT: linalg.matmul
// CHECK-NEXT: linalg.sub
// CHECK-NEXT: linalg.add
// CHECK-NEXT: return

// -----

!type = tensor<2048x2048xf32>
func.func @expect_no_fold_as_orig_dest_not_additive_zero(%arg0: !type, %arg1: !type) -> !type {
  %0 = arith.constant dense<1.111111e+00> : !type
  %cst = arith.constant 0.000000e+00 : f32
  %1 = tensor.empty() : !type
  %2 = linalg.fill ins(%cst : f32) outs(%1 : !type) -> !type
  %3 = linalg.matmul ins(%arg0, %0 : !type, !type) outs(%2 : !type) -> !type
  %4 = linalg.matmul ins(%arg1, %0 : !type, !type) outs(%0 : !type) -> !type
  %5 = linalg.add ins(%3, %4 : !type, !type) outs(%1 : !type) -> !type
  return %5 : !type
}

// CHECK-LABEL: func.func @expect_no_fold_as_orig_dest_not_additive_zero
// CHECK: linalg.fill
// CHECK-NEXT: linalg.matmul
// CHECK-NEXT: linalg.matmul
// CHECK-NEXT: linalg.add
// CHECK-NEXT: return

// -----

!type = tensor<2048x2048xf32>
func.func @expect_no_fold_as_contraction_result_has_multiple_users(%arg0: !type, %arg1: !type) -> (!type, !type) {
  %0 = arith.constant dense<1.111111e+00> : !type
  %cst = arith.constant 0.000000e+00 : f32
  %1 = tensor.empty() : !type
  %2 = linalg.fill ins(%cst : f32) outs(%1 : !type) -> !type
  %3 = linalg.matmul ins(%arg0, %0 : !type, !type) outs(%2 : !type) -> !type
  %4 = linalg.matmul ins(%arg1, %0 : !type, !type) outs(%0 : !type) -> !type
  %5 = linalg.add ins(%3, %4 : !type, !type) outs(%1 : !type) -> !type
  %6 = linalg.mul ins(%4, %arg0 : !type, !type) outs(%1 : !type) -> !type
  return %5, %6 : !type, !type
}

// CHECK-LABEL: func.func @expect_no_fold_as_contraction_result_has_multiple_users
// CHECK: linalg.fill
// CHECK-NEXT: linalg.matmul
// CHECK-NEXT: linalg.matmul
// CHECK-NEXT: linalg.add
// CHECK-NEXT: linalg.mul
// CHECK-NEXT: return


// -----

#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d0)>  // NB: not an ordered projection

!type = tensor<2048x2048xf32>
func.func @expect_no_fold_as_dest_accumulation_is_not_identity_mapped(%arg0: !type, %arg1: !type) -> !type {
  %0 = arith.constant dense<1.111111e+00> : !type
  %cst = arith.constant 0.000000e+00 : f32
  %1 = tensor.empty() : !type
  %2 = linalg.fill ins(%cst : f32) outs(%1 : !type) -> !type
  %3 = linalg.generic { indexing_maps = [#map0, #map1, #map2],
                        iterator_types = ["parallel", "parallel", "reduction"] }
    ins(%arg0, %0: !type, !type) outs(%2: !type) {
      ^bb0(%a: f32, %b: f32, %c: f32):
        %5 = arith.mulf %a, %b : f32
        %6 = arith.addf %c, %5 : f32
        linalg.yield %6 : f32
  } -> !type
  %4 = linalg.add ins(%3, %arg1 : !type, !type) outs(%1 : !type) -> !type
  return %4 : !type
}

// CHECK-LABEL: func.func @expect_no_fold_as_dest_accumulation_is_not_identity_mapped
// CHECK: linalg.fill
// CHECK-NEXT: linalg.generic
// CHECK: linalg.add
// CHECK-NEXT: return

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>  // NB: is an ordered projection

!type = tensor<2048x2048xf32>
func.func @expect_add_to_fold(%arg0: !type, %arg1: !type) -> !type {
  %0 = arith.constant dense<1.111111e+00> : !type
  %cst = arith.constant 0.000000e+00 : f32
  %1 = tensor.empty() : !type
  %2 = linalg.fill ins(%cst : f32) outs(%1 : !type) -> !type
  %3 = linalg.generic { indexing_maps = [#map0, #map1, #map2],
                        iterator_types = ["parallel", "parallel", "reduction"] }
    ins(%arg0, %0: !type, !type) outs(%2: !type) {
      ^bb0(%a: f32, %b: f32, %c: f32):
        %5 = arith.mulf %a, %b : f32
        %6 = arith.addf %c, %5 : f32
        linalg.yield %6 : f32
  } -> !type
  %4 = linalg.add ins(%3, %arg1 : !type, !type) outs(%1 : !type) -> !type
  return %4 : !type
}

// CHECK-LABEL: func.func @expect_add_to_fold
// CHECK: linalg.generic
// CHECK-NOT: linalg.add
// CHECK: return