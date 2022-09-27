// RUN: standalone-opt %s -convert-xsmm-to-func="use-extract-metadata" | FileCheck %s

// CHECK-DAG: func.func private @xsmm_brgemm_dispatch_f32(i64, i64, i64, i64, i64, i64) -> i64
// CHECK-DAG: func.func private @xsmm_brgemm_invoke_f32(i64, !llvm.ptr<f32>, index, index, index, index, index, index, index, !llvm.ptr<f32>, index, index, index, index, index, index, index, !llvm.ptr<f32>, index, index, index, index, index, i64)
func.func @dispatch_brgemm(%arg0: memref<2x5x4xf32>, %arg1: memref<2x4x5xf32>,
                           %arg2: memref<4x4xf32>) -> memref<4x4xf32> {
  %0 = xsmm.ternary.dispatch brgemm [5, 5, 4, 4, 5, 5] (dataType f32)
  %c2_i64 = arith.constant 2 : i64
  xsmm.ternary brgemm(%0, %arg0, %arg1, %arg2, %c2_i64) : (i64, memref<2x5x4xf32>, memref<2x4x5xf32>, memref<4x4xf32>, i64) -> ()
  return %arg2 : memref<4x4xf32>
}
