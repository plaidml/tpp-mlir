// RUN: standalone-opt %s -convert-tpp-to-xsmm -split-input-file | FileCheck %s

func.func @identity_to_xsmm(%arg0: memref<3x3xf32>) {
  // CHECK: todo
  %cst = arith.constant 0.000000e+00 : f32
  tpp.identity ins(%cst: f32) out(%arg0: memref<3x3xf32>)
  return 
}

// -----

func.func @identity_to_xsmm(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) {
  // CHECK: todo
  tpp.identity ins(%arg0: memref<3x3xf32>) out(%arg1: memref<3x3xf32>)
  return 
}
