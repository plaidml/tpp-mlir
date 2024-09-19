
// Test 1
// gemm f32
// RUN: tpp-opt %s --tile-linalg="mTile=4,8 nTile=8,16" | FileCheck %s

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


// Test 2
// Chained gemm f32
// RUN: tpp-opt %s --tile-linalg="mTile=4,8 nTile=8,16" | FileCheck %s

module {
  memref.global "private" constant @__constant_48x32x32xf32 : memref<48x32x32xf32> = dense<1.000000e+00> {alignment = 64 : i64}
  func.func @entry(%arg0: memref<8x48x32x32xf32>) -> memref<8x48x32x32xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = memref.get_global @__constant_48x32x32xf32 : memref<48x32x32xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x48x32x32xf32>
    scf.forall (%arg1, %arg2) in (8, 48) {
      %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
      linalg.fill ins(%cst : f32) outs(%subview : memref<32x32xf32, strided<[32, 1], offset: ?>>)
      %subview_1 = memref.subview %arg0[%arg1, 0, 0, 0] [1, 48, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>
      linalg.batch_reduce_matmul ins(%subview_1, %0 : memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<48x32x32xf32>) outs(%subview : memref<32x32xf32, strided<[32, 1], offset: ?>>)
    }
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<8x48x32x32xf32>
    scf.forall (%arg1, %arg2) in (8, 48) {
      %subview = memref.subview %alloc_0[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
      linalg.fill ins(%cst : f32) outs(%subview : memref<32x32xf32, strided<[32, 1], offset: ?>>)
      %subview_1 = memref.subview %alloc[%arg1, 0, 0, 0] [1, 48, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>
      linalg.batch_reduce_matmul ins(%subview_1, %0 : memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<48x32x32xf32>) outs(%subview : memref<32x32xf32, strided<[32, 1], offset: ?>>)
    }
    scf.forall (%arg1, %arg2) in (8, 48) {
      %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
      linalg.fill ins(%cst : f32) outs(%subview : memref<32x32xf32, strided<[32, 1], offset: ?>>)
      %subview_1 = memref.subview %alloc_0[%arg1, 0, 0, 0] [1, 48, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>
      linalg.batch_reduce_matmul ins(%subview_1, %0 : memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<48x32x32xf32>) outs(%subview : memref<32x32xf32, strided<[32, 1], offset: ?>>)
    }
    return %alloc : memref<8x48x32x32xf32>
  }
}
