// RUN: tpp-opt %s -convert-tpp-to-xsmm -split-input-file | FileCheck %s

// CHECK-LABEL: relu_to_xsmm(
// CHECK-SAME:  %[[ARG0:.+]]: memref<5x6xf32>)
func.func @relu_to_xsmm(%arg0: memref<5x6xf32>) {
  // CHECK: %[[DISPACTH:.+]] = xsmm.unary.dispatch relu [5, 6, 6, 6] flags = (none) data_type = f32
  // CHECK-NEXT: xsmm.unary relu(data_type = f32, %[[DISPACTH]], %[[ARG0]], %[[ARG0]])
  tpp.relu ins(%arg0: memref<5x6xf32>) outs(%arg0: memref<5x6xf32>)
  return
}

// -----

// CHECK-LABEL: @relu_to_xsmm(
// CHECK-SAME:  %[[ARG0:.+]]: memref<5x6xf32>, %[[ARG1:.+]]: memref<5x6xf32>)
func.func @relu_to_xsmm(%arg0: memref<5x6xf32>, %arg1: memref<5x6xf32>) {
  // CHECK: %[[DISPACTH:.+]] = xsmm.unary.dispatch relu [5, 6, 6, 6] flags = (none) data_type = f32
  // CHECK-NEXT: xsmm.unary relu(data_type = f32, %[[DISPACTH]], %[[ARG0]], %[[ARG1]])
  tpp.relu ins(%arg0: memref<5x6xf32>) outs(%arg1: memref<5x6xf32>)
  return
}

// -----

// CHECK-LABEL: relu_to_xsmm(
// CHECK-SAME:  %[[ARG0:.+]]: memref<1x32xf32>)
func.func @relu_to_xsmm(%arg0: memref<1x32xf32>) {
  // CHECK: %[[DISPATCH:.+]] = xsmm.unary.dispatch relu [1, 32, 32, 32] flags = (none) data_type = f32
  // CHECK-NEXT: xsmm.unary relu(data_type = f32, %[[DISPATCH]], %[[ARG0]], %[[ARG0]])
  tpp.relu ins(%arg0: memref<1x32xf32>) outs(%arg0: memref<1x32xf32>)
  return
}

// -----

// CHECK-LABEL: relu_to_xsmm
// CHECK-SAME: %[[ARG0:.+]]: memref<3x3xf32>, %[[ARG1:.+]]: f32
func.func @relu_to_xsmm(%arg0: memref<3x3xf32>, %arg1: f32) {
  // CHECK: %[[DIS:.+]] = xsmm.unary.dispatch relu [3, 3, 1, 3] flags = (bcast_scalar) data_type = f32
  // CHECK: xsmm.unary relu(data_type = f32, %[[DIS]], %[[ARG1]], %[[ARG0]])
  tpp.relu ins(%arg1: f32) outs(%arg0: memref<3x3xf32>)
  return
}

// -----

// CHECK-LABEL: relu_to_xsmm
// CHECK-SAME: %[[ARG0:.+]]: memref<3x3xf32>, %[[ARG1:.+]]: memref<1xf32>
func.func @relu_to_xsmm(%arg0: memref<3x3xf32>, %arg1: memref<1xf32>) {
  // CHECK: %[[DIS:.+]] = xsmm.unary.dispatch relu [3, 3, 1, 3] flags = (bcast_scalar) data_type = f32
  // CHECK: xsmm.unary relu(data_type = f32, %[[DIS]], %[[ARG1]], %[[ARG0]])
  tpp.relu ins(%arg1: memref<1xf32>) outs(%arg0: memref<3x3xf32>)
  return
}

// -----

// CHECK-LABEL: relu_to_xsmm
// CHECK-SAME: %[[ARG0:.+]]: memref<3x3xf32>, %[[ARG1:.+]]: memref<1x1xf32>
func.func @relu_to_xsmm(%arg0: memref<3x3xf32>, %arg1: memref<1x1xf32>) {
  // CHECK: %[[DIS:.+]] = xsmm.unary.dispatch relu [3, 3, 1, 3] flags = (bcast_scalar) data_type = f32
  // CHECK: xsmm.unary relu(data_type = f32, %[[DIS]], %[[ARG1]], %[[ARG0]])
  tpp.relu ins(%arg1: memref<1x1xf32>) outs(%arg0: memref<3x3xf32>)
  return
}
