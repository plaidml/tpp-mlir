// RUN: standalone-opt %s -convert-tpp-to-xsmm -split-input-file | FileCheck %s

// CHECK-LABEL: @identity_to_xsmm(
// CHECK-SAME: %[[arg:.*]]: memref<3x3xf32>) {
func.func @identity_to_xsmm(%arg0: memref<3x3xf32>) {
  // CHECK: %[[cst:.*]] = arith.constant 0.000000e+00 : f32
  %cst = arith.constant 0.000000e+00 : f32
  // CHECK: xsmm.unary @xsmm_identity_invoke(%[[cst]], %[[arg]]) : (f32, memref<3x3xf32>) -> ()
  tpp.identity ins(%cst: f32) out(%arg0: memref<3x3xf32>)
  return 
}

// -----

// CHECK-LABEL: @identity_to_xsmm(
// CHECK-SAME: %[[arg_zero:.*]]: memref<3x3xf32>, %[[arg_one:.*]]: memref<3x3xf32>)
func.func @identity_to_xsmm(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) {
  // xsmm.unary @xsmm_identity_invoke(%[[arg_zero]], %[[arg_one]]) : (memref<3x3xf32>, memref<3x3xf32>) -> ()
  tpp.identity ins(%arg0: memref<3x3xf32>) out(%arg1: memref<3x3xf32>)
  return 
}

// -----

// CHECK-LABEL: @matmul_to_xsmm(
// CHECK-SAME: %[[arg_zero:.*]]: memref<3x3xf32>, %[[arg_one:.*]]: memref<3x3xf32>, %[[arg_two:.*]]: memref<3x3xf32>)
func.func @matmul_to_xsmm(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>, %arg2: memref<3x3xf32>) {
  // CHECK: %[[cst:.*]] = arith.constant 3 : i32
  // CHECK: %[[dispatch:.*]] = xsmm.dispatch @xsmm_matmul_dispatch
  // CHECK: xsmm.ternary @xsmm_matmul_invoke(%[[dispatch]], %[[arg_zero]], %[[arg_one]], %[[arg_two]]) : (i64, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>) -> ()
  tpp.matmul ins(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) out(%arg2: memref<3x3xf32>)
  return 
}
