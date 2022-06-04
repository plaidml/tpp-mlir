// RUN: standalone-opt %s | standalone-opt | FileCheck %s

// CHECK-LABEL: func.func @bar
func.func @bar(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>) {
  return
}
