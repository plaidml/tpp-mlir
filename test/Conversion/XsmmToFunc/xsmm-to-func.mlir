// RUN: tpp-opt %s -convert-xsmm-to-func -split-input-file | FileCheck %s

// CHECK-LABEL: dispatch_unary
func.func @dispatch_unary() -> i64 {
  %0 = xsmm.unary.dispatch identity [5, 6, 5, 6](broadcast row dataType f32)
  return %0: i64
}

// -----

// CHECK-LABEL: dispatch_brgemm
func.func @dispatch_brgemm() -> i64 {
  %0 = xsmm.brgemm.dispatch [5, 5, 4, 4, 5, 5] (flags = none, data_type = f32)
  return %0 : i64
}

// -----

// CHECK-LABEL: dispatch_gemm
func.func @dispatch_gemm(%arg0: memref<3x3xf32>) -> i64 {
  %0 = xsmm.matmul.dispatch [1, 2, 3, 4, 5, 6] (flags = none, data_type = f32)
  return %0 : i64
}
