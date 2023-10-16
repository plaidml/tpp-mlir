// RUN: tpp-opt %s -default-tpp-passes="linalg-to-xsmm" -split-input-file | FileCheck %s

func.func @fill_op(%arg0: memref<3x3xf32>) {
  %cst = arith.constant 0.0 : f32
  linalg.fill ins(%cst : f32) outs(%arg0 : memref<3x3xf32>)
  return
}

// CHECK-LABEL: fill_op
// CHECK-SAME: %[[ARG0:.+]]: memref<3x3xf32>
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : i64
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i64
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : i64
// CHECK-DAG: %[[C8:.+]] = arith.constant 8 : i64
// CHECK-DAG: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[DIS:.+]] = call @xsmm_unary_dispatch(%[[C2]], %[[C1]], %[[C3]], %[[C3]], %[[C1]], %[[C3]], %[[C8]])
// CHECK: %[[PTR:.+]] = memref.extract_aligned_pointer_as_index %[[ARG0]] : memref<3x3xf32> -> index
// CHECK: %[[PTR_TO_INT:.+]] = arith.index_cast %[[PTR]] : index to i64
// CHECK: %[[LLVM_PTR:.+]] = llvm.inttoptr %[[PTR_TO_INT]] : i64 to !llvm.ptr<f32>
// CHECK: call @xsmm_unary_scalar_invoke(%[[C1]], %[[DIS]], %[[CST]], %[[LLVM_PTR]], %[[C0]])

// -----

func.func @fill_op_i32(%arg0: memref<3x3xi32>) {
  %cst = arith.constant 0 : i32
  linalg.fill ins(%cst : i32) outs(%arg0 : memref<3x3xi32>)
  return
}

// CHECK-LABEL: fill_op_i32
// CHECK-NOT: xsmm
