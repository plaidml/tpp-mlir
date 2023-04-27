// RUN: tpp-opt %s -convert-tpp-to-xsmm -split-input-file | FileCheck %s

// CHECK-LABEL: @zero_to_xsmm(
// CHECK-SAME:  %[[ARG0:.+]]: memref<5x6xf32>)
func.func @zero_to_xsmm(%arg0: memref<5x6xf32>) {
  // CHECK: %[[DISPACTH:.+]] = xsmm.unary.dispatch zero [5, 6, 6, 6] flags = (none) data_type = f32
  // CHECK-NEXT: xsmm.unary zero(data_type = f32, %[[DISPACTH]], %[[ARG0]], %[[ARG0]])
  tpp.zero ins(%arg0: memref<5x6xf32>) outs(%arg0: memref<5x6xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @zero_to_xsmm(
// CHECK-SAME:  %[[ARG0:.+]]: memref<32xf32>)
func.func @zero_to_xsmm(%arg0: memref<32xf32>) {
  // CHECK: %[[DISPATCH:.+]] = xsmm.unary.dispatch zero [1, 32, 32, 32] flags = (none) data_type = f32
  // CHECK-NEXT: xsmm.unary zero(data_type = f32, %[[DISPATCH]], %[[ARG0]], %[[ARG0]])
  tpp.zero ins(%arg0: memref<32xf32>) outs(%arg0: memref<32xf32>)
  return
}
