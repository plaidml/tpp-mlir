// RUN: tpp-opt %s -split-input-file -verify-diagnostics

// CHECK-LABEL: func.func @xsmm_dialect
func.func @xsmm_dialect(%arg0: memref<2x2xf32>) -> memref<2x2xf32> {
  return %arg0: memref<2x2xf32>
}
