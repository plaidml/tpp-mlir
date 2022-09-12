// RUN: standalone-opt %s -convert-tpp-to-xsmm -convert-xsmm-to-func -split-input-file | FileCheck %s

// CHECK: func.func private @xsmm_matmul_invoke_f32(i64, memref<*xf32>, memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
// CHECK: func.func private @xsmm_matmul_dispatch_f32(i64, i64, i64, i64, i64, i64) -> i64 attributes {llvm.emit_c_interface}

// CHECK-LABEL: func.func @tpp_matmul(
func.func @tpp_matmul(%arg0: memref<3x6xf32>, %arg1: memref<6x3xf32>, %arg2: memref<3x3xf32>) {
  // CHECK: call @xsmm_matmul_dispatch
  // CHECK: memref.cast
  // CHECK: memref.cast
  // CHECK: memref.cast
  // CHECK: call @xsmm_matmul_invoke
  tpp.matmul ins(%arg0: memref<3x6xf32>, %arg1: memref<6x3xf32>) out(%arg2: memref<3x3xf32>)
  return 
}
