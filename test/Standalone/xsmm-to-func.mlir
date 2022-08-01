// RUN: standalone-opt %s -convert-xsmm-to-func -split-input-file | FileCheck %s

// CHECK: func.func private @xsmm_matmul_dispatch(i64, i64, i64, i64, i64, i64) -> i64 attributes {llvm.emit_c_interface}
// CHECK: func.func private @xsmm_unary_dispatch(i64, i64, i64, i64, i64, i64) -> i64 attributes {llvm.emit_c_interface}
func.func @dispatch_matmul(%arg0: memref<32x256xf32>, %arg1: memref<1x8x32x32xf32>) -> i64 {
  %0 = xsmm.unary.dispatch identity [5, 6, 5, 6](bcast_row)
  %1 = xsmm.ternary.dispatch matmul [3, 3, 3, 3, 3, 3]
  %2 = arith.addi %0, %1 : i64

  xsmm.copy ins(%arg0: memref<32x256xf32>) outs(%arg1: memref<1x8x32x32xf32>) [1, 8, 32, 32]

  return %2: i64
}

// -----

// CHECK: func.func private @xsmm_brgemm_dispatch(i64, i64, i64, i64, i64, i64) -> i64 attributes {llvm.emit_c_interface}
func.func @dispatch_brgemm(%arg0: memref<2x5x4xf32>, %arg1: memref<2x4x5xf32>,
                           %arg2: memref<4x4xf32>) -> memref<4x4xf32> {
  %0 = xsmm.ternary.dispatch brgemm [5, 5, 4, 4, 5, 5]
  %c2_i64 = arith.constant 2 : i64
  xsmm.ternary brgemm(%0, %arg0, %arg1, %arg2, %c2_i64) : (i64, memref<2x5x4xf32>, memref<2x4x5xf32>, memref<4x4xf32>, i64) -> ()
  return %arg2 : memref<4x4xf32>
}
