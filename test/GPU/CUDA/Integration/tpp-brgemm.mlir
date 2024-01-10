// RUN: ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0:${ASAN_OPTIONS} \
// RUN: tpp-run %s -gpu=cuda \
// RUN:  -entry-point-result=void -e entry 2>&1 | \
// RUN: FileCheck %s

func.func @entry(%arg0: memref<2x8x32x32xf32>, %arg1: memref<8x8x32x32xf32>, %arg2: memref<2x8x32x32xf32>) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c2, %c8) step (%c1, %c1) {
    %subview = memref.subview %arg0[%arg3, 0, 0, 0] [1, 8, 32, 32] [1, 1, 1, 1] 
      : memref<2x8x32x32xf32> to memref<8x32x32xf32, strided<[1024, 32, 1], offset: ?>>
    %subview_0 = memref.subview %arg1[%arg4, 0, 0, 0] [1, 8, 32, 32] [1, 1, 1, 1] 
      : memref<8x8x32x32xf32> to memref<8x32x32xf32, strided<[1024, 32, 1], offset: ?>>
    %subview_1 = memref.subview %arg2[%arg3, %arg4, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] 
      : memref<2x8x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
    linalg.batch_reduce_matmul ins(%subview, %subview_0 : 
                                   memref<8x32x32xf32, strided<[1024, 32, 1], offset: ?>>, 
                                   memref<8x32x32xf32, strided<[1024, 32, 1], offset: ?>>) 
                 outs(%subview_1 : memref<32x32xf32, strided<[32, 1], offset: ?>>)
    scf.reduce
  }

  %out = memref.alloc() : memref<2x8x32x32xf32>
  %tw = gpu.wait async
  %tOut = gpu.memcpy async [%tw] %out, %arg2 : memref<2x8x32x32xf32>, memref<2x8x32x32xf32>
  gpu.wait [%tOut]

  %d1 = arith.constant -1.0 : f32
  %zeroCst = arith.constant 0 : index
  %v0 = vector.transfer_read %out[%zeroCst, %zeroCst, %zeroCst, %zeroCst], %d1 : memref<2x8x32x32xf32>, vector<32xf32>
  vector.print %v0 : vector<32xf32>

  memref.dealloc %out : memref<2x8x32x32xf32>

  return
}

// CHECK: 257, 257, 257, 257, 257, 257, 257, 257, 257, 257, 257, 257, 257, 257, 257, 257, 257, 257, 257, 257, 257, 257, 257, 257, 257, 257, 257, 257, 257, 257, 257, 257
