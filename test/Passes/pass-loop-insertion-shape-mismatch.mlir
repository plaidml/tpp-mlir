// RUN: tpp-opt --loop-insertion-pass="M-tile-shape=2 N-tile-shape=8" --canonicalize --split-input-file %s -debug-only=loop-insertion 2>&1 | FileCheck %s

module{
        func.func @tiling_shape_mismatch(%arg0: memref<8x4x4x4xbf16>, %arg1: memref<4x4x2x4x2xbf16>, %arg2: memref<8x4x4x4xbf16>) {
          %c4_i64 = arith.constant 4 : i64
          scf.forall (%arg3, %arg4) in (8, 4) {
            %subview = memref.subview %arg0[%arg3, 0, 0, 0] [1, 4, 4, 4] [1, 1, 1, 1] : memref<8x4x4x4xbf16> to memref<4x4x4xbf16, strided<[16, 4, 1], offset: ?>>
            %subview_0 = memref.subview %arg1[%arg4, 0, 0, 0, 0] [1, 4, 2, 4, 2] [1, 1, 1, 1, 1] : memref<4x4x2x4x2xbf16> to memref<4x2x4x2xbf16, strided<[16, 8, 2, 1], offset: ?>>
            %subview_1 = memref.subview %arg2[%arg3, %arg4, 0, 0] [1, 1, 4, 4] [1, 1, 1, 1] : memref<8x4x4x4xbf16> to memref<4x4xbf16, strided<[4, 1], offset: ?>>
            %0 = xsmm.brgemm.dispatch [4, 4, 4, 4, 4, 4, 16, 16] flags = (vnni_b) data_type = bf16
            xsmm.brgemm(data_type = bf16, %0, %subview, %subview_0, %subview_1, %c4_i64) : (i64, memref<4x4x4xbf16, strided<[16, 4, 1], offset: ?>>, memref<4x2x4x2xbf16, strided<[16, 8, 2, 1], offset: ?>>, memref<4x4xbf16, strided<[4, 1], offset: ?>>, i64) -> ()
          }
          return
        }
}

// CHECK: Failed to tile the loop
