// RUN: standalone-opt %s -split-input-file -verify-diagnostics

func.func @myfunc(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>) -> memref<2x2xf32> {

  // expected-error @below {{Expect single operand}}
  xsmm.unary_call @libxsmm_relu(%arg0, %arg1)
    : (memref<2x2xf32>, memref<2x2xf32>) -> () 
  return %arg1: memref<2x2xf32>
}
