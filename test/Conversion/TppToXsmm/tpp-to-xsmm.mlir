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

// CHECK-LABEL: @matmul_to_xsmm(
// CHECK-SAME: %[[ARG0:.*]]: memref<3x3xf32>, %[[ARG1:.*]]: memref<3x3xf32>, %[[ARG2:.*]]: memref<3x3xf32>)
func.func @matmul_to_xsmm(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>, %arg2: memref<3x3xf32>) {
  // CHECK: %[[DISPATCH:.*]] = xsmm.ternary.dispatch matmul [3, 3, 3, 3, 3, 3](dataType f32 isVNNI false)
  // CHECK-NEXT: xsmm.ternary matmul(dataType f32, %[[DISPATCH]], %[[ARG0]], %[[ARG1]], %[[ARG2]]) 
  tpp.matmul ins(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) out(%arg2: memref<3x3xf32>)
  return
}

// -----

// CHECK-LABEL: @identity_to_xsmm(
// CHECK-SAME:  %[[ARG0:.+]]: f32,
// CHECK-SAME:  %[[ARG1:.+]]: memref<5x6xf32>)
func.func @identity_to_xsmm(%arg0: f32, %arg1: memref<5x6xf32>) {

  // CHECK: %[[DISPATCH:.+]] = xsmm.unary.dispatch identity [5, 6, 1, 6](broadcast scalar dataType f32)
  // CHECK-NEXT: xsmm.unary identity(dataType f32, %[[DISPATCH]], %[[ARG0]], %[[ARG1]]) 
  tpp.identity ins(%arg0: f32) out(%arg1: memref<5x6xf32>)
  return
}

// -----

// CHECK-LABEL: @identity_to_xsmm(
// CHECK-SAME:  %[[ARG0:.+]]: memref<1x1xf32>, %[[ARG1:.+]]: memref<5x6xf32>)
func.func @identity_to_xsmm(%arg0: memref<1x1xf32>, %arg1: memref<5x6xf32>) {
  // CHECK: %[[DISPATCH:.+]] = xsmm.unary.dispatch identity [5, 6, 1, 6](broadcast scalar dataType f32)
  // CHECK-NEXT: xsmm.unary identity(dataType f32, %[[DISPATCH]], %[[ARG0]], %[[ARG1]]) 
  tpp.identity ins(%arg0: memref<1x1xf32>) out(%arg1: memref<5x6xf32>)
  return
}

// -----

// CHECK-LABEL: @relu_to_xsmm(
// CHECK-SAME:  %[[ARG0:.+]]: memref<5x6xf32>)
func.func @relu_to_xsmm(%arg0: memref<5x6xf32>) {
  // CHECK: %[[DISPACTH:.+]] = xsmm.unary.dispatch relu [5, 6, 6, 6](broadcast none dataType f32)
  // CHECK-NEXT: xsmm.unary relu(dataType f32, %[[DISPACTH]], %[[ARG0]], %[[ARG0]])
  tpp.relu ins(%arg0: memref<5x6xf32>) out(%arg0: memref<5x6xf32>)
  return
}

// -----

// CHECK-LABEL: @brgemm_to_xsmm(
// CHECK-SAME:  %[[ARG0:.+]]: memref<3x5x4xf32>, %[[ARG1:.+]]: memref<3x4x5xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: memref<5x5xf32>)
func.func @brgemm_to_xsmm(%arg0: memref<3x5x4xf32>, %arg1: memref<3x4x5xf32>,
                          %arg2: memref<5x5xf32>) -> memref<5x5xf32> {
  // CHECK: %[[BATCH:.+]] = arith.constant 3 : i64
  // CHECK-NEXT: %[[DISPATCH:.+]] = xsmm.ternary.dispatch brgemm [5, 5, 4, 4, 5, 5](dataType f32 isVNNI false)
  // CHECK-NEXT: xsmm.ternary brgemm(dataType f32, %[[DISPATCH]], %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[BATCH]])
  tpp.brgemm ins(%arg0: memref<3x5x4xf32>, %arg1: memref<3x4x5xf32>)
             out(%arg2: memref<5x5xf32>)
  return %arg2: memref<5x5xf32>
}

// -----

// Strides are non-constant expect to fail.
func.func @tpp_matmul(%arg0: memref<12x9xf32, strided<[?, ?], offset: ?>>,
                      %arg1: memref<9x6xf32, strided<[?, ?], offset: ?>>,
                      %arg2: memref<12x6xf32, strided<[?, ?], offset: ?>>) {
  // CHECK-NOT: xsmm.ternary matmul
  tpp.matmul ins(%arg0 : memref<12x9xf32, strided<[?, ?], offset: ?>>,
                 %arg1 : memref<9x6xf32, strided<[?, ?], offset: ?>>)
             out(%arg2 : memref<12x6xf32, strided<[?, ?], offset: ?>>)
  return
}

// -----

// CHECK-LABEL: @relu_to_xsmm(
// CHECK-SAME:  %[[ARG0:.+]]: memref<5x6xf32>, %[[ARG1:.+]]: memref<5x6xf32>)
func.func @relu_to_xsmm(%arg0: memref<5x6xf32>, %arg1: memref<5x6xf32>) {
  // CHECK: %[[DISPACTH:.+]] = xsmm.unary.dispatch relu [5, 6, 6, 6](broadcast none dataType f32)
  // CHECK-NEXT: xsmm.unary relu(dataType f32, %[[DISPACTH]], %[[ARG0]], %[[ARG1]])
  tpp.relu ins(%arg0: memref<5x6xf32>) out(%arg1: memref<5x6xf32>)
  return
}

// -----

// CHECK-LABEL: func.func @add_to_xsmm(
// CHECK-SAME:  %[[ARG0:.+]]: memref<32xf32>, %[[ARG1:.+]]: memref<32xf32>, 
// CHECK-SAME:  %[[ARG2:.+]]: memref<32xf32>)
func.func @add_to_xsmm(%arg0: memref<32xf32>, %arg1: memref<32xf32>, %arg2: memref<32xf32>) {
  // CHECK: %[[DISPATCH:.+]] = xsmm.binary.dispatch add [1, 32, 1, 1, 1](broadcast none dataType f32)
  // CHECK-NEXT: xsmm.binary add(dataType f32, %[[DISPATCH]], %[[ARG0]], %[[ARG1]], %[[ARG2]])
  tpp.add ins(%arg0: memref<32xf32>, %arg1: memref<32xf32>) out(%arg2: memref<32xf32>)
  return 
}

// -----

// CHECK-LABEL: func.func @relu_to_xsmm(
// CHECK-SAME:  %[[ARG0:.+]]: memref<32xf32>)
func.func @relu_to_xsmm(%arg0: memref<32xf32>) {
  // CHECK: %[[DISPATCH:.+]] = xsmm.unary.dispatch relu [1, 32, 1, 1](broadcast none dataType f32)
  // CHECK-NEXT: xsmm.unary relu(dataType f32, %[[DISPATCH]], %[[ARG0]], %[[ARG0]])
  tpp.relu ins(%arg0: memref<32xf32>) out(%arg0: memref<32xf32>)
  return
}
