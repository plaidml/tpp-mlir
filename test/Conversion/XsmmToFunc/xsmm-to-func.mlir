// RUN: tpp-opt %s -convert-xsmm-to-func -split-input-file | FileCheck %s

// CHECK-LABEL: dispatch_unary
func.func @dispatch_unary() -> i64 {
  %0 = xsmm.unary.dispatch identity [5, 6, 5, 6](broadcast row dataType f32)
  return %0: i64
}

// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i64
// CHECK-DAG: %[[C5:.+]] = arith.constant 5 : i64
// CHECK-DAG: %[[C6:.+]] = arith.constant 6 : i64
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : i64
// CHECK: call @xsmm_unary_dispatch(%[[C1]], %[[C5]], %[[C6]], %[[C5]], %[[C6]], %[[C1]], %[[C2]])

// -----

// CHECK-LABEL: dispatch_brgemm
func.func @dispatch_brgemm() -> i64 {
  %0 = xsmm.brgemm.dispatch [5, 5, 4, 4, 5, 5] (flags = none, data_type = f32)
  return %0 : i64
}

// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i64
// CHECK-DAG: %[[C5:.+]] = arith.constant 5 : i64
// CHECK-DAG: %[[C4:.+]] = arith.constant 4 : i64
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i64
// CHECK: call @xsmm_brgemm_dispatch(%[[C1]], %[[C5]], %[[C5]], %[[C4]], %[[C4]], %[[C5]], %[[C5]], %[[C0]])

// -----

// CHECK-LABEL: dispatch_gemm
func.func @dispatch_gemm(%arg0: memref<3x3xf32>) -> i64 {
  %0 = xsmm.matmul.dispatch [1, 2, 3, 4, 5, 6] (flags = none, data_type = f32)
  return %0 : i64
}

// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i64
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : i64
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : i64
// CHECK-DAG: %[[C4:.+]] = arith.constant 4 : i64
// CHECK-DAG: %[[C5:.+]] = arith.constant 5 : i64
// CHECK-DAG: %[[C6:.+]] = arith.constant 6 : i64
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i64
// CHECK: call @xsmm_matmul_dispatch(%[[C1]], %[[C1]], %[[C2]], %[[C3]], %[[C4]], %[[C5]], %[[C6]], %[[C0]])
