// RUN: tpp-opt %s -convert-tpp-to-xsmm -split-input-file | FileCheck %s

// CHECK-LABEL: @identity_to_xsmm(
// CHECK-SAME: %[[ARG0:.+]]: memref<3x3xf32>) {
func.func @identity_to_xsmm(%arg0: memref<3x3xf32>) {
  // CHECK: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
  %cst = arith.constant 0.000000e+00 : f32

  // m = 3
  // n = 3
  // ldi = 1
  // ldo = 3
  // input_type = 1 (F32)
  // output_type = 1 (F32)
  // compute_type = 1 (F32)
  // b_cast = 3 (bcast scalar)

  // CHECK: %[[DISPATCH:.+]] = xsmm.unary.dispatch identity [3, 3, 1, 3](broadcast scalar dataType f32)
  // CHECK: xsmm.unary identity(dataType f32, %[[DISPATCH]], %[[CST]], %[[ARG0]]) 
  tpp.identity ins(%cst: f32) out(%arg0: memref<3x3xf32>)
  return
}

// -----

// CHECK-LABEL: @identity_to_xsmm(
// CHECK-SAME: %[[ARG0:.+]]: memref<3x3xf32>, %[[ARG1:.*]]: memref<3x3xf32>)
func.func @identity_to_xsmm(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) {

  // m = 3
  // n = 3
  // ldi = 3
  // ldo = 3
  // input_type = 1 (F32)
  // output_type = 1 (F32)
  // compute_type = 1 (F32)
  // b_cast = 0 (bcast none)

  // CHECK: %[[DISPATCH:.+]] = xsmm.unary.dispatch identity [3, 3, 3, 3](broadcast none dataType f32)
  // CHECK: xsmm.unary identity(dataType f32, %[[DISPATCH]], %[[ARG0]], %[[ARG1]]) 
  tpp.identity ins(%arg0: memref<3x3xf32>) out(%arg1: memref<3x3xf32>)
  return
}

// -----

// CHECK-LABEL: @identity_to_xsmm(
// CHECK-SAME:  %[[ARG0:.+]]: memref<5x1xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<5x6xf32>)
func.func @identity_to_xsmm(%arg0: memref<5x1xf32>, %arg1: memref<5x6xf32>) {

  // m = 5
  // n = 6
  // ldi = 1
  // ldo = 6
  // input_type = 1 (F32)
  // output_type = 1 (F32)
  // compute_type = 1 (F32)
  // b_cast = 1 (bcast row)

  // CHECK: xsmm.unary.dispatch identity [5, 6, 1, 6](broadcast row dataType f32)
  // CHECK: xsmm.unary identity
  tpp.identity ins(%arg0: memref<5x1xf32>) out(%arg1: memref<5x6xf32>)
  return
}

// -----

// CHECK-LABEL: @identity_to_xsmm(
// CHECK-SAME:  %[[ARG0:.+]]: memref<1x5xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<5x5xf32>)
func.func @identity_to_xsmm(%arg0: memref<1x5xf32>, %arg1: memref<5x5xf32>) {
  
  // m = 5
  // n = 5
  // ldi = 5
  // ldo = 5
  // input_type = 1 (F32)
  // output_type = 1 (F32)
  // compute_type = 1 (F32)
  // b_cast = 2 (bcast col)

  // CHECK: %[[DISPATCH:.+]] = xsmm.unary.dispatch identity [5, 5, 5, 5](broadcast col dataType f32)
  // CHECK-NEXT: xsmm.unary identity(dataType f32, %[[DISPATCH]], %[[ARG0]], %[[ARG1]]) 
  tpp.identity ins(%arg0: memref<1x5xf32>) out(%arg1: memref<5x5xf32>)
  return
}

// -----

// CHECK-LABEL: @identity_to_xsmm(
// CHECK-SAME:  %[[ARG0:.+]]: f32,
// CHECK-SAME:  %[[ARG1:.+]]: memref<5x6xf32>)
func.func @identity_to_xsmm(%arg0: f32, %arg1: memref<5x6xf32>) {
  // m = 5
  // n = 6
  // ldi = 1
  // ldo = 6
  // CHECK: %[[DISPATCH:.+]] = xsmm.unary.dispatch identity [5, 6, 1, 6](broadcast scalar dataType f32)
  // CHECK-NEXT: xsmm.unary identity(dataType f32, %[[DISPATCH]], %[[ARG0]], %[[ARG1]]) 
  tpp.identity ins(%arg0: f32) out(%arg1: memref<5x6xf32>)
  return
}

// -----

// CHECK-LABEL: @identity_to_xsmm(
// CHECK-SAME:  %[[ARG0:.+]]: memref<1x1xf32>, %[[ARG1:.+]]: memref<5x6xf32>)
func.func @identity_to_xsmm(%arg0: memref<1x1xf32>, %arg1: memref<5x6xf32>) {
  // m = 5
  // n = 6
  // ldi = 1
  // ldo = 6
  // CHECK: %[[DISPATCH:.+]] = xsmm.unary.dispatch identity [5, 6, 1, 6](broadcast scalar dataType f32)
  // CHECK-NEXT: xsmm.unary identity(dataType f32, %[[DISPATCH]], %[[ARG0]], %[[ARG1]]) 
  tpp.identity ins(%arg0: memref<1x1xf32>) out(%arg1: memref<5x6xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @identity_1d
// CHECK-SAME: %[[ARG0:.+]]: memref<8xf32, strided<[1]>>, %[[ARG1:.+]]: memref<8xf32, strided<[1]>>
func.func @identity_1d(%arg0: memref<8xf32, strided<[1]>>, %arg1: memref<8xf32, strided<[1]>>) {
  // m = 1
  // n = 8
  // ldi = 8
  // ldo = 8
  // CHECK: %[[DISPATCH:.+]] = xsmm.unary.dispatch identity [1, 8, 8, 8](broadcast none dataType f32)
  // CHECK-NEXT: xsmm.unary identity(dataType f32, %[[DISPATCH]], %[[ARG0]], %[[ARG1]])
  tpp.identity ins(%arg0 : memref<8xf32, strided<[1]>>) out(%arg1 : memref<8xf32, strided<[1]>>)
  return
}

// -----

// CHECK-LABEL: func.func @identity_1d
// CHECK-SAME: %[[ARG0:.+]]: memref<1xf32, strided<[1]>>, %[[ARG1:.+]]: memref<8xf32, strided<[1]>>
func.func @identity_1d(%arg0: memref<1xf32, strided<[1]>>, %arg1: memref<8xf32, strided<[1]>>) {
  // m = 1
  // n = 8
  // ldi = 1
  // ldo = 8
  // CHECK: %[[DISPATCH:.+]] = xsmm.unary.dispatch identity [1, 8, 1, 8](broadcast scalar dataType f32)
  // CHECK-NEXT: xsmm.unary identity(dataType f32, %[[DISPATCH]], %[[ARG0]], %[[ARG1]])
  tpp.identity ins(%arg0 : memref<1xf32, strided<[1]>>) out(%arg1 : memref<8xf32, strided<[1]>>)
  return
}
