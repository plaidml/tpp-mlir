// RUN: standalone-opt %s -convert-xsmm-to-func -split-input-file | FileCheck %s

// CHECK: func.func private @xsmm_matmul_dispatch(i64, i64, i64, i64, i64, i64) -> i64 attributes {llvm.emit_c_interface}
// CHECK: func.func private @xsmm_unary_dispatch(i64, i64, i64, i64, i64, i64) -> i64 attributes {llvm.emit_c_interface}
func.func @dispatch_matmul() -> i64 {
  %0 = xsmm.unary.dispatch identity [5, 6, 5, 6](bcast_row)
  %1 = xsmm.ternary.dispatch matmul [3, 3, 3, 3, 3, 3]
  %2 = arith.addi %0, %1 : i64
  return %2: i64
}
