// RUN: tpp-opt %s -convert-xsmm-to-func -split-input-file | FileCheck %s

// CHECK-LABEL: dispatch_unary
func.func @dispatch_unary() -> i64 {
  %0 = xsmm.unary.dispatch identity [5, 6, 5, 6] flags = (bcast_row) data_type = f32
  return %0: i64
}

// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i64
// CHECK-DAG: %[[C5:.+]] = arith.constant 5 : i64
// CHECK-DAG: %[[C6:.+]] = arith.constant 6 : i64
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : i64
// CHECK: call @xsmm_unary_dispatch(%[[C1]], %[[C1]], %[[C5]], %[[C6]], %[[C5]], %[[C6]], %[[C2]])

// -----

// CHECK-LABEL: dispatch_brgemm
func.func @dispatch_brgemm() -> i64 {
  %0 = xsmm.brgemm.dispatch [5, 5, 4, 4, 5, 5] flags = (none) data_type = f32
  return %0 : i64
}

// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i64
// CHECK-DAG: %[[C5:.+]] = arith.constant 5 : i64
// CHECK-DAG: %[[C4:.+]] = arith.constant 4 : i64
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i64
// CHECK: call @xsmm_brgemm_dispatch(%[[C1]], %[[C5]], %[[C5]], %[[C4]], %[[C4]], %[[C5]], %[[C5]], %[[C0]])

// -----

// CHECK-LABEL: dispatch_gemm
func.func @dispatch_gemm() -> i64 {
  %0 = xsmm.gemm.dispatch [1, 2, 3, 4, 5, 6] flags = (none) data_type = f32
  return %0 : i64
}

// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i64
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : i64
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : i64
// CHECK-DAG: %[[C4:.+]] = arith.constant 4 : i64
// CHECK-DAG: %[[C5:.+]] = arith.constant 5 : i64
// CHECK-DAG: %[[C6:.+]] = arith.constant 6 : i64
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i64
// CHECK: call @xsmm_gemm_dispatch(%[[C1]], %[[C1]], %[[C2]], %[[C3]], %[[C4]], %[[C5]], %[[C6]], %[[C0]])

// -----

// CHECK-LABEL: dispatch_gemm
func.func @dispatch_gemm() -> i64 {
  %0 = xsmm.gemm.dispatch [1, 2, 3, 4, 5, 6] flags = (vnni_a, vnni_b) data_type = bf16
  return %0 : i64
}

// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i64
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : i64
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : i64
// CHECK-DAG: %[[C4:.+]] = arith.constant 4 : i64
// CHECK-DAG: %[[C5:.+]] = arith.constant 5 : i64
// CHECK-DAG: %[[C6:.+]] = arith.constant 6 : i64
// Or between 2048 and 4096 (see enum for GemmFlags)
// CHECK-DAG: %[[C6144:.+]] = arith.constant 6144 : i64
// CHECK: call @xsmm_gemm_dispatch(%[[C2]], %[[C1]], %[[C2]], %[[C3]], %[[C4]], %[[C5]], %[[C6]], %[[C6144]])

// -----

// CHECK-LABEL: dispatch_gemm
func.func @dispatch_gemm() -> i64 {
  %0 = xsmm.gemm.dispatch [1, 2, 3, 4, 5, 6] flags = (vnni_a, vnni_b, vnni_c) data_type = bf16
  return %0 : i64
}

// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i64
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : i64
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : i64
// CHECK-DAG: %[[C4:.+]] = arith.constant 4 : i64
// CHECK-DAG: %[[C5:.+]] = arith.constant 5 : i64
// CHECK-DAG: %[[C6:.+]] = arith.constant 6 : i64
// Or between 2048 and 4096 and 8192 (see enum for GemmFlags)
// CHECK-DAG: %[[C14336:.+]] = arith.constant 14336 : i64
// CHECK: call @xsmm_gemm_dispatch(%[[C2]], %[[C1]], %[[C2]], %[[C3]], %[[C4]], %[[C5]], %[[C6]], %[[C14336]])

// -----

// CHECK-LABEL: dispatch_gemm
func.func @dispatch_gemm() -> i64 {
  %0 = xsmm.gemm.dispatch [1, 2, 3, 4, 5, 6] flags = (vnni_a) data_type = bf16
  return %0 : i64
}

// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i64
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : i64
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : i64
// CHECK-DAG: %[[C4:.+]] = arith.constant 4 : i64
// CHECK-DAG: %[[C5:.+]] = arith.constant 5 : i64
// CHECK-DAG: %[[C6:.+]] = arith.constant 6 : i64
// LIBXSMM is col-major check we swap the flag for A and B (see enum for GemmFlags)
// CHECK-DAG: %[[C4096:.+]] = arith.constant 4096 : i64
// CHECK: call @xsmm_gemm_dispatch(%[[C2]], %[[C1]], %[[C2]], %[[C3]], %[[C4]], %[[C5]], %[[C6]], %[[C4096]])

// -----

// CHECK-LABEL: dispatch_gemm
func.func @dispatch_gemm() -> i64 {
  %0 = xsmm.gemm.dispatch [1, 2, 3, 4, 5, 6] flags = (vnni_b) data_type = bf16
  return %0 : i64
}

// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i64
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : i64
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : i64
// CHECK-DAG: %[[C4:.+]] = arith.constant 4 : i64
// CHECK-DAG: %[[C5:.+]] = arith.constant 5 : i64
// CHECK-DAG: %[[C6:.+]] = arith.constant 6 : i64
// LIBXSMM is col-major check we swap the flag for A and B (see enum for GemmFlags)
// CHECK-DAG: %[[C2048:.+]] = arith.constant 2048 : i64
// CHECK: call @xsmm_gemm_dispatch(%[[C2]], %[[C1]], %[[C2]], %[[C3]], %[[C4]], %[[C5]], %[[C6]], %[[C2048]])
