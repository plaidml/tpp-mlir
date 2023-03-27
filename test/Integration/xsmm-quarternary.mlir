// RUN: tpp-run %s -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

func.func @entry(%arg0: memref<2x3x4xf32>, %arg1: memref<2x4x3xf32>, %arg2: memref<3xbf16>, %arg3: memref<3x3xf32>) {
  %c16_i64 = arith.constant 16 : i64
  %func = xsmm.quarternary.dispatch fused_brgemm [3, 3, 4, 4, 3, 3](dataType f32, isVNNI false)
  xsmm.quarternary fused_brgemm(dataType bf16, %func, %arg0, %arg1, %arg2, %arg3, %c16_i64) : (i64, memref<2x3x4xf32>, memref<2x4x3xf32>, memref<3xbf16>, memref<3x3xf32>, i64) -> ()

  return
}

// CHECK: ( 1, 1, 1 )
// CHECK: ( 1, 1, 1 )
// CHECK: ( 1, 1, 1 )
