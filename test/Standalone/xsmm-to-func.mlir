// RUN: standalone-opt %s -convert-xsmm-to-func -split-input-file | FileCheck %s

func.func @dispatch_matmul() {
  // CHECK: func.func private @xsmm_matmul_dispatch(i32) -> i64 attributes {llvm.emit_c_interface}
  %c3_i32 = arith.constant 3 : i32
  %0 = xsmm.dispatch @xsmm_matmul_dispatch(%c3_i32) : (i32) -> i64
  return 
}
