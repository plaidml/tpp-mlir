// RUN: standalone-opt %s -convert-tpp-to-xsmm -split-input-file | FileCheck %s

// CHECK-LABEL: @identity_to_xsmm(
// CHECK-SAME: %[[arg:.*]]: memref<3x3xf32>) {
func.func @identity_to_xsmm(%arg0: memref<3x3xf32>) {
  // CHECK: %[[cst:.*]] = arith.constant 0.000000e+00 : f32
  %cst = arith.constant 0.000000e+00 : f32

  // m = 3
  // n = 3
  // ldi = 1
  // ldo = 3
  // input_type = 1 (F32)
  // output_type = 1 (F32)
  // compute_type = 1 (F32)
  // b_cast = 3 (bcast scalar)
 
  // CHECK: xsmm.unary.dispatch identity [3, 3, 1, 3](bcast_scalar)
  // CHECK: xsmm.unary identity
  tpp.identity ins(%cst: f32) out(%arg0: memref<3x3xf32>)
  return 
}

// -----

// CHECK-LABEL: @identity_to_xsmm(
// CHECK-SAME: %[[arg_zero:.*]]: memref<3x3xf32>, %[[arg_one:.*]]: memref<3x3xf32>)
func.func @identity_to_xsmm(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) {

  // m = 3
  // n = 3
  // ldi = 3
  // ldo = 3
  // input_type = 1 (F32)
  // output_type = 1 (F32)
  // compute_type = 1 (F32)
  // b_cast = 0 (bcast none)

  // CHECK: xsmm.unary.dispatch identity [3, 3, 3, 3](none)
  // CHECK: xsmm.unary identity
  tpp.identity ins(%arg0: memref<3x3xf32>) out(%arg1: memref<3x3xf32>)
  return 
}

// -----

// CHECK-LABEL: @identity_to_xsmm(
func.func @identity_to_xsmm(%arg0: memref<5x1xf32>, %arg1: memref<5x6xf32>) {

  // m = 5
  // n = 6
  // ldi = 1
  // ldo = 6
  // input_type = 1 (F32)
  // output_type = 1 (F32)
  // compute_type = 1 (F32)
  // b_cast = 1 (bcast row)

  // CHECK: xsmm.unary.dispatch identity [5, 6, 1, 6](bcast_row)
  // CHECK: xsmm.unary identity
  tpp.identity ins(%arg0: memref<5x1xf32>) out(%arg1: memref<5x6xf32>)
  return 
}

// -----

// CHECK-LABEL: @identity_to_xsmm(
func.func @identity_to_xsmm(%arg0: memref<1x5xf32>, %arg1: memref<5x5xf32>) {

  // m = 5
  // n = 5
  // ldi = 5
  // ldo = 5
  // input_type = 1 (F32)
  // output_type = 1 (F32)
  // compute_type = 1 (F32)
  // b_cast = 2 (bcast col)

  // CHECK: xsmm.unary.dispatch identity [5, 5, 5, 5](bcast_col)
  // CHECK: xsmm.unary identity
  tpp.identity ins(%arg0: memref<1x5xf32>) out(%arg1: memref<5x5xf32>)
  return 
}

// -----

// CHECK-LABEL: @matmul_to_xsmm(
// CHECK-SAME: %[[arg_zero:.*]]: memref<3x3xf32>, %[[arg_one:.*]]: memref<3x3xf32>, %[[arg_two:.*]]: memref<3x3xf32>)
func.func @matmul_to_xsmm(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>, %arg2: memref<3x3xf32>) {
  // CHECK: %[[dispatch:.*]] = xsmm.ternary.dispatch matmul [3, 3, 3, 3, 3, 3]
  // CHECK: xsmm.ternary matmul(%[[dispatch]], %[[arg_zero]], %[[arg_one]], %[[arg_two]]) : (i64, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>) -> ()
  tpp.matmul ins(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) out(%arg2: memref<3x3xf32>)
  return 
}

// -----

// CHECK-LABEL: @identity_to_xsmm(
func.func @identity_to_xsmm(%arg0: f32, %arg1: memref<5x6xf32>) {

  // CHECK: xsmm.unary.dispatch identity [5, 6, 1, 6](bcast_scalar)
  // CHECK: xsmm.unary identity
  tpp.identity ins(%arg0: f32) out(%arg1: memref<5x6xf32>)
  return
}

// -----

// CHECK-LABEL: @identity_to_xsmm(
func.func @identity_to_xsmm(%arg0: memref<1x1xf32>, %arg1: memref<5x6xf32>) {

  // CHECK: xsmm.unary.dispatch identity [5, 6, 1, 6](bcast_scalar)
  // CHECK: xsmm.unary identity 
  tpp.identity ins(%arg0: memref<1x1xf32>) out(%arg1: memref<5x6xf32>)
  return
}

// -----

// CHECK-LABEL: @relu_to_xsmm(
func.func @relu_to_xsmm(%arg0: memref<5x6xf32>, %arg1: memref<5x6xf32>) {

  // CHECK: xsmm.unary.dispatch relu [5, 6, 6, 6](none)
  // CHECK: xsmm.unary relu
  tpp.relu ins(%arg0: memref<5x6xf32>) out(%arg1: memref<5x6xf32>)
  return
}

// -----

// CHECK-LABEL: @brgemm_to_xsmm(
func.func @brgemm_to_xsmm(%arg0: memref<3x5x4xf32>, %arg1: memref<3x4x5xf32>,
                          %arg2: memref<5x5xf32>) -> memref<5x5xf32> {
  // CHECK: xsmm.ternary.dispatch brgemm [5, 5, 4, 4, 5, 5]
  // CHECK: xsmm.ternary brgemm 
  tpp.brgemm ins(%arg0: memref<3x5x4xf32>, %arg1: memref<3x4x5xf32>)
             out(%arg2: memref<5x5xf32>)
  return %arg2: memref<5x5xf32>
}
