// RUN: tpp-run %s -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

func.func @entry(%arg0: memref<64x4x4xbf16>, %arg1: memref<64x2x4x2xbf16>,
                              %arg2: memref<4x4xbf16>) {
  %c64_i64 = arith.constant 64 : i64
  %0 = xsmm.ternary.dispatch brgemm [4, 4, 4, 4, 4, 4](dataType bf16, isVNNI true)
  xsmm.ternary brgemm(dataType bf16, %0, %arg0, %arg1, %arg2, %c64_i64) : (i64, memref<64x4x4xbf16>, memref<64x2x4x2xbf16>, memref<4x4xbf16>, i64) -> ()

  return
}

// CHECK: ( 256, 256, 256, 256 )
// CHECK: ( 256, 256, 256, 256 )
// CHECK: ( 256, 256, 256, 256 )
// CHECK: ( 256, 256, 256, 256 )
