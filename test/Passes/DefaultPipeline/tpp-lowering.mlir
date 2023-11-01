// RUN: tpp-opt %s -tpp-lowering="linalg-to-xsmm=false" | FileCheck %s -check-prefix=XSMM
// RUN: tpp-opt %s -tpp-lowering="tpp-to-loops" | FileCheck %s -check-prefix=LOOPS

func.func @tpp_ops(%arg0: memref<3x5x4xf32>, %arg1: memref<3x4x5xf32>, %arg2: memref<5x5xf32>, %arg3: memref<5x5xf32>) {
    tpp.brgemm ins(%arg0 : memref<3x5x4xf32>, %arg1 : memref<3x4x5xf32>, %arg2 : memref<5x5xf32>) 
               outs(%arg2 : memref<5x5xf32>)
    tpp.relu ins(%arg2 : memref<5x5xf32>) outs(%arg2 : memref<5x5xf32>)
    tpp.gemm ins(%arg2 : memref<5x5xf32>, %arg3 : memref<5x5xf32>, %arg2 : memref<5x5xf32>) 
             outs(%arg2 : memref<5x5xf32>)
    return
  }

// XSMM-LABEL: func.func @tpp_ops(
// XSMM-NOT: tpp.brgemm
// XSMM: xsmm.brgemm
// XSMM-NOT: tpp.relu
// XSMM: xsmm.unary relu
// XSMM-NOT: tpp.gemm
// XSMM: xsmm.gemm

// LOOPS-LABEL: func.func @tpp_ops(
// LOOPS-NOT: tpp.brgemm
// LOOPS: scf.for
// LOOPS:   arith.mulf
// LOOPS:   arith.addf
// LOOPS-NOT: tpp.relu
// LOOPS: scf.for
// LOOPS:   arith.maximumf
// LOOPS-NOT: tpp.gemm
// LOOPS: scf.for
// LOOPS:   arith.mulf
// LOOPS:   arith.addf

// XSMM-LABEL: copy
func.func @copy(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>) {
  // XSMM: xsmm.unary.dispatch identity
  // XSMM-NEXT: xsmm.unary identity
  memref.copy %arg0, %arg1 : memref<2x2xf32> to memref<2x2xf32>
  return
}
