// RUN: tpp-opt %s -tpp-lowering -split-input-file | FileCheck %s -check-prefix=XSMM
// RUN: tpp-opt %s -tpp-lowering="tpp-to-loops" -split-input-file | FileCheck %s -check-prefix=LOOPS

func.func @tpp_ops(%arg0: memref<3x5x4xf32>, %arg1: memref<3x4x5xf32>, %arg2: memref<5x5xf32>, %arg3: memref<5x5xf32>) {
    tpp.brgemm ins(%arg0 : memref<3x5x4xf32>, %arg1 : memref<3x4x5xf32>) out(%arg2 : memref<5x5xf32>)
    tpp.relu ins(%arg2 : memref<5x5xf32>) out(%arg2 : memref<5x5xf32>)
    tpp.matmul ins(%arg2 : memref<5x5xf32>, %arg3 : memref<5x5xf32>) out(%arg2 : memref<5x5xf32>)
    return
  }

// XSMM-LABEL: func.func @tpp_ops(
// XSMM-NOT: tpp.brgemm
// XSMM: xsmm.ternary brgemm
// XSMM-NOT: tpp.relu
// XSMM: xsmm.unary relu
// XSMM-NOT: tpp.matmul
// XSMM: xsmm.ternary matmul

// LOOPS-LABEL: func.func @tpp_ops(
// LOOPS-NOT: tpp.brgemm
// LOOPS: scf.for
// LOOPS:   arith.mulf
// LOOPS:   arith.addf
// LOOPS-NOT: tpp.relu
// LOOPS: scf.for
// LOOPS:   arith.maxf
// LOOPS-NOT: tpp.matmul
// LOOPS: scf.for
// LOOPS:   arith.mulf
// LOOPS:   arith.addf
