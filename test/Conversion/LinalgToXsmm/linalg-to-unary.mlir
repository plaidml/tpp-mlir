// RUN: tpp-opt %s -convert-linalg-to-xsmm -split-input-file | FileCheck %s

func.func @fill_op(%arg0: memref<32x32xf32>) {
  %cst = arith.constant 0.0 : f32
  linalg.fill ins(%cst : f32) outs(%arg0 : memref<32x32xf32>)
  return
}

// CHECK-LABEL: fill_op
// CHECK-SAME: %[[ARG0:.+]]: memref<32x32xf32>
// CHECK: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[DIS:.+]] = xsmm.unary.dispatch zero [32, 32, 1, 32] flags = (bcast_scalar) data_type = f32
// CHECK: xsmm.unary zero(data_type = f32, %[[DIS]], %[[CST]], %[[ARG0]])

// -----

func.func @fill_op(%arg0: memref<32x32xf32>, %cst: f32) {
  linalg.fill ins(%cst : f32) outs(%arg0 : memref<32x32xf32>)
  return
}

// CHECK-LABEL: fill_op
// CHECK: linalg.fill
// CHECK-NOT: xsmm.unary

// -----

func.func @fill_op(%arg0: memref<32x32xbf16>) {
  %cst = arith.constant 0.0 : bf16
  linalg.fill ins(%cst : bf16) outs(%arg0 : memref<32x32xbf16>)
  return
}

// CHECK-LABEL: fill_op
// CHECK-SAME: %[[ARG0:.+]]: memref<32x32xbf16>
// CHECK: %[[CST:.+]] = arith.constant 0.000000e+00 : bf16
// CHECK: %[[DIS:.+]] = xsmm.unary.dispatch zero [32, 32, 1, 32] flags = (bcast_scalar) data_type = bf16
// CHECK: xsmm.unary zero(data_type = bf16, %[[DIS]], %[[CST]], %[[ARG0]])

// -----

func.func @fill_op(%arg0: memref<32xf32>) {
  %cst = arith.constant 0.0 : f32
  linalg.fill ins(%cst : f32) outs(%arg0 : memref<32xf32>)
  return
}

// CHECK-LABEL: fill_op
// CHECK: linalg.fill
// CHECK-NOT: xsmm.unary
