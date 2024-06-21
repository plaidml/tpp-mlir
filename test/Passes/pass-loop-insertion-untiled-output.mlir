// RUN: tpp-opt --loop-insertion-pass="M-tile-shape=2 N-tile-shape=2" --canonicalize --split-input-file %s --verify-diagnostics

#map = affine_map<(d0) -> (d0 * 32)>
func.func @entry(%arg0: memref<32x32x32x32xbf16>, %arg1: memref<32x32x16x32x2xbf16>, %arg2: memref<1024x1024xbf16>) {
  %c32_i64 = arith.constant 32 : i64
  // @expected-warning @below {{Failed to tile the loop}}
  scf.forall (%arg3, %arg4) in (32, 32) {
    %0 = affine.apply #map(%arg3)
    %1 = affine.apply #map(%arg4)
    %subview = memref.subview %arg0[%arg3, 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<32x32x32x32xbf16> to memref<32x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
    %subview_0 = memref.subview %arg1[%arg4, 0, 0, 0, 0] [1, 32, 16, 32, 2] [1, 1, 1, 1, 1] : memref<32x32x16x32x2xbf16> to memref<32x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
    %subview_1 = memref.subview %arg2[%0, %1] [32, 32] [1, 1] : memref<1024x1024xbf16> to memref<32x32xbf16, strided<[1024, 1], offset: ?>>
    %2 = xsmm.brgemm.dispatch [32, 32, 32, 32, 32, 1024, 1024, 1024] flags = (vnni_b) data_type = bf16
    xsmm.brgemm(data_type = bf16, %2, %subview, %subview_0, %subview_1, %c32_i64) : (i64, memref<32x32x32xbf16, strided<[1024, 32, 1], offset: ?>>, memref<32x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>, memref<32x32xbf16, strided<[1024, 1], offset: ?>>, i64) -> ()
  }
  return
}
