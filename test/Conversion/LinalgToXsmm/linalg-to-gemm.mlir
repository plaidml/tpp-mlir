// RUN: tpp-opt %s -convert-linalg-to-xsmm | FileCheck %s

func.func @simple_gemm(%arg0: memref<32x64xf32, strided<[64, 1], offset: ?>>,
                       %arg1: memref<64x32xf32, strided<[32, 1], offset: ?>>,
                       %arg2: memref<32x32xf32, strided<[32, 1], offset: ?>>) {
  linalg.matmul ins(%arg0, %arg1 : memref<32x64xf32, strided<[64, 1], offset: ?>>, 
                                   memref<64x32xf32, strided<[32, 1], offset: ?>>) 
                outs(%arg2 : memref<32x32xf32, strided<[32, 1], offset: ?>>)
  return
}

// CHECK-LABEL: simple_gemm
// CHECK-SAME: %[[ARG0:.+]]: memref<32x64xf32, strided<[64, 1], offset: ?>>, 
// CHECK-SAME: %[[ARG1:.+]]: memref<64x32xf32, strided<[32, 1], offset: ?>>, 
// CHECK-SAME: %[[ARG2:.+]]: memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK: %[[DIS:.+]] = xsmm.gemm.dispatch [32, 32, 64, 64, 32, 32] flags = (none) data_type = f32
// CHECK: xsmm.gemm(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]], %[[ARG2]])
