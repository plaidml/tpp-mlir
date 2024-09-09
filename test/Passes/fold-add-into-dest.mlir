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
func.func @expect_add_to_not_fold(%arg0: !type, %arg1: !type) -> !type {
  %0 = arith.constant dense<1.111111e+00> : !type
  %cst = arith.constant 0.000000e+00 : f32
  %1 = tensor.empty() : !type
  %2 = linalg.fill ins(%cst : f32) outs(%1 : !type) -> !type
  %3 = linalg.matmul_transpose_b ins(%arg0, %0 : !type, !type) outs(%2 : !type) -> !type
  %4 = linalg.add ins(%3, %3 : !type, !type) outs(%1 : !type) -> !type
  return %4 : !type
}

// CHECK-LABEL: func.func @expect_add_to_not_fold
// CHECK: linalg.fill
// CHECK-NEXT: linalg.matmul_transpose_b
// CHECK-NEXT: linalg.add
// CHECK-NEXT: return

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
