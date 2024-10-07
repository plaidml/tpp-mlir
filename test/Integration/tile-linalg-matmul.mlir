// RUN: tpp-opt %s  | tpp-run -e entry --entry-point-result=void -print > %t.1
// RUN: tpp-opt %s  --tile-linalg="mTile=4,8 nTile=8,16"  | tpp-run -e entry --entry-point-result=void -print > %t.2
// RUN: diff %t.1 %t.2 | FileCheck %s --check-prefix=DIFF --allow-empty

module {
  func.func @entry(%arg0: memref<16x32x16x32xf32>, %arg1: memref<32x32x32x32xf32>, %arg2: memref<16x32x16x32xf32>) {
    scf.forall (%arg3, %arg4) in (16, 32) {
      %subview = memref.subview %arg0[%arg3, 0, 0, 0] [1, 32, 16, 32] [1, 1, 1, 1] : memref<16x32x16x32xf32> to memref<32x16x32xf32, strided<[512, 32, 1], offset: ?>>
      %subview_0 = memref.subview %arg1[%arg4, 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<32x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
      %subview_1 = memref.subview %arg2[%arg3, %arg4, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<16x32x16x32xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
      linalg.batch_reduce_matmul ins(%subview, %subview_0 : memref<32x16x32xf32, strided<[512, 32, 1], offset: ?>>, memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>) outs(%subview_1 : memref<16x32xf32, strided<[32, 1], offset: ?>>)
    }
    return
  }
}
