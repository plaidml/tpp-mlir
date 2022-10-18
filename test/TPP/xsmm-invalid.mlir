// RUN: tpp-opt %s -split-input-file -verify-diagnostics

// CHECK-LABEL: func.func @myfunc
func.func @myfunc(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>) -> memref<2x2xf32> {
  return %arg0: memref<2x2xf32>
}
