// RUN: tpp-opt %s -convert-tpp-to-xsmm -convert-xsmm-to-func -split-input-file | FileCheck %s

// CHECK: func.func private @xsmm_gemm_invoke(i64, i64, memref<*xf32>, memref<*xf32>, memref<*xf32>) attributes {llvm.emit_c_interface}
// CHECK: func.func private @xsmm_gemm_dispatch(i64, i64, i64, i64, i64, i64, i64, i64) -> i64 attributes {llvm.emit_c_interface}

// CHECK-LABEL: func.func @tpp_gemm(
func.func @tpp_gemm(%arg0: memref<3x6xf32>, %arg1: memref<6x3xf32>, %arg2: memref<3x3xf32>) {
  // CHECK: call @xsmm_gemm_dispatch
  // CHECK: memref.cast
  // CHECK: memref.cast
  // CHECK: memref.cast
  // CHECK: call @xsmm_gemm_invoke
  tpp.gemm ins(%arg0: memref<3x6xf32>, %arg1: memref<6x3xf32>, %arg2: memref<3x3xf32>) 
           outs(%arg2: memref<3x3xf32>)
  return
}
