// RUN: tpp-opt %s -convert-check-to-func | FileCheck %s

// CHECK: func.func private @expect_almost_equals(tensor<4x4xf32>, tensor<4x4xf32>, f32) attributes {llvm.emit_c_interface}
// CHECK: func.func private @expect_true(i1) attributes {llvm.emit_c_interface}
func.func @myfunc(%arg0:tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: f32){
  %a = arith.constant 1:i1
  // CHECK: call @expect_true(%true) : (i1) -> ()
  check.expect_true(%a):i1

 // CHECK: call @expect_almost_equals(%arg0, %arg1, %arg2) : (tensor<4x4xf32>, tensor<4x4xf32>, f32) -> ()
 check.expect_almost_eq(%arg0, %arg1, %arg2) : tensor<4x4xf32>, tensor<4x4xf32>, f32
 return
}
