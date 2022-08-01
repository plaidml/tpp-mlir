// RUN: standalone-opt %s -convert-linalg-to-tpp | FileCheck %s

// CHECK-LABEL: func.func @brgemmLowering
func.func @brgemmLowering(%arg0: memref<3x5x4xf32>, %arg1: memref<3x4x5xf32>, 
                          %arg2: memref<5x5xf32>) -> memref<5x5xf32> {
  // CHECK: tpp.brgemm
  linalg.reduce_batch_matmul ins(%arg0, %arg1: memref<3x5x4xf32>, memref<3x4x5xf32>)
                             outs(%arg2: memref<5x5xf32>)
  return %arg2: memref<5x5xf32>
}
