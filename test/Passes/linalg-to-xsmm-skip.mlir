// RUN: tpp-opt %s -convert-linalg-to-xsmm -split-input-file | FileCheck %s
// RUN: tpp-opt %s -convert-linalg-to-xsmm="skip-operations=fill" -split-input-file | FileCheck %s --check-prefix=SKIP-FILL
// RUN: tpp-opt %s -convert-linalg-to-xsmm="skip-operations=matmul" -split-input-file | FileCheck %s --check-prefix=SKIP-GEMM

func.func @fill_op(%arg0: memref<32x32xf32>) {
  %cst = arith.constant 0.0 : f32
  linalg.fill ins(%cst : f32) outs(%arg0 : memref<32x32xf32>)
  return
}

// CHECK-LABEL: fill_op
// CHECK: xsmm.unary
// SKIP-FILL: linalg.fill
// SKIP-GEMM: xsmm.unary

// -----

func.func @simple_gemm(%arg0: memref<32x64xf32, strided<[64, 1], offset: ?>>,
                       %arg1: memref<64x32xf32, strided<[32, 1], offset: ?>>,
                       %arg2: memref<32x32xf32, strided<[32, 1], offset: ?>>) {
  linalg.matmul ins(%arg0, %arg1 : memref<32x64xf32, strided<[64, 1], offset: ?>>,
                                   memref<64x32xf32, strided<[32, 1], offset: ?>>)
                outs(%arg2 : memref<32x32xf32, strided<[32, 1], offset: ?>>)
  return
}


// CHECK-LABEL: simple_gemm
// CHECK: xsmm.gemm
// SKIP-FILL: xsmm.gemm
// SKIP-GEMM: linalg.matmul
