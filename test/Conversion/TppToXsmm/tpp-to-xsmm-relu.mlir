// RUN: tpp-opt %s -convert-tpp-to-xsmm -split-input-file | FileCheck %s

// CHECK-LABEL: @relu_to_xsmm(
// CHECK-SAME:  %[[ARG0:.+]]: memref<5x6xf32>)
func.func @relu_to_xsmm(%arg0: memref<5x6xf32>) {
  // CHECK: %[[DISPACTH:.+]] = xsmm.unary.dispatch relu [5, 6, 6, 6] flags = (none) data_type = f32
  // CHECK-NEXT: xsmm.unary relu(dataType f32, %[[DISPACTH]], %[[ARG0]], %[[ARG0]])
  tpp.relu ins(%arg0: memref<5x6xf32>) outs(%arg0: memref<5x6xf32>)
  return
}

// -----

// CHECK-LABEL: @relu_to_xsmm(
// CHECK-SAME:  %[[ARG0:.+]]: memref<5x6xf32>, %[[ARG1:.+]]: memref<5x6xf32>)
func.func @relu_to_xsmm(%arg0: memref<5x6xf32>, %arg1: memref<5x6xf32>) {
  // CHECK: %[[DISPACTH:.+]] = xsmm.unary.dispatch relu [5, 6, 6, 6] flags = (none) data_type = f32
  // CHECK-NEXT: xsmm.unary relu(dataType f32, %[[DISPACTH]], %[[ARG0]], %[[ARG1]])
  tpp.relu ins(%arg0: memref<5x6xf32>) outs(%arg1: memref<5x6xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @relu_to_xsmm(
// CHECK-SAME:  %[[ARG0:.+]]: memref<32xf32>)
func.func @relu_to_xsmm(%arg0: memref<32xf32>) {
  // CHECK: %[[DISPATCH:.+]] = xsmm.unary.dispatch relu [1, 32, 32, 32] flags = (none) data_type = f32
  // CHECK-NEXT: xsmm.unary relu(dataType f32, %[[DISPATCH]], %[[ARG0]], %[[ARG0]])
  tpp.relu ins(%arg0: memref<32xf32>) outs(%arg0: memref<32xf32>)
  return
}
