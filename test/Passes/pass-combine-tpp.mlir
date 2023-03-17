//RUN: tpp-opt %s --combine-tpp | FileCheck %s

func.func @mlp(%arg0: memref<8x16x32x32xbf16>, %arg1: memref<16x16x16x32x2xbf16>, %arg2: memref<512xbf16>, %arg3: memref<8x16x32x32xbf16>) -> memref<8x16x32x32xbf16> {
  %cst = arith.constant -1.000000e+00 : bf16
  %c0 = arith.constant 0 : index
  %expand_shape = memref.expand_shape %arg2 [[0, 1]] : memref<512xbf16> into memref<16x32xbf16>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x16x32x32xbf16>
  scf.forall (%arg4, %arg5) in (8, 16) {
    %subview_0 = memref.subview %arg0[%arg4, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<8x16x32x32xbf16> to memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
    %subview_1 = memref.subview %arg1[%arg5, 0, 0, 0, 0] [1, 16, 16, 32, 2] [1, 1, 1, 1, 1] : memref<16x16x16x32x2xbf16> to memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
    %subview_2 = memref.subview %arg3[%arg4, %arg5, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x16x32x32xbf16> to memref<32x32xbf16, strided<[32, 1], offset: ?>>
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<32x32xbf16>
    memref.copy %subview_2, %alloc_3 : memref<32x32xbf16, strided<[32, 1], offset: ?>> to memref<32x32xbf16>
    %cast = memref.cast %alloc_3 : memref<32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>

    //CHECK: tpp.fused_vnni_brgemm ins(%{{.*}} : memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>>, %{{.*}} : memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>, %{{.*}} : memref<32xbf16, strided<[1], offset: ?>>) out(%{{.*}} : memref<32x32xbf16, strided<[32, 1], offset: ?>>)
    tpp.vnni_brgemm ins(%subview_0 : memref<16x32x32xbf16, strided<[1024, 32, 1], offset: ?>>, %subview_1 : memref<16x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>) out(%cast : memref<32x32xbf16, strided<[32, 1]>>)
    %subview_4 = memref.subview %expand_shape[%arg5, 0] [1, 32] [1, 1] : memref<16x32xbf16> to memref<32xbf16, strided<[1], offset: ?>>
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xbf16>
    tpp.add_bcast ins(%alloc_3 : memref<32x32xbf16>, %subview_4 : memref<32xbf16, strided<[1], offset: ?>>) out(%alloc_5 : memref<32x32xbf16>)
    %subview_6 = memref.subview %alloc[%arg4, %arg5, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x16x32x32xbf16> to memref<32x32xbf16, strided<[32, 1], offset: ?>>
    tpp.relu ins(%alloc_5 : memref<32x32xbf16>) out(%subview_6 : memref<32x32xbf16, strided<[32, 1], offset: ?>>)
    memref.dealloc %alloc_3 : memref<32x32xbf16>
    memref.dealloc %alloc_5 : memref<32x32xbf16>
  }
  %subview = memref.subview %alloc[0, 0, 0, 0] [1, 1, 4, 4] [1, 1, 1, 1] : memref<8x16x32x32xbf16> to memref<4x4xbf16, strided<[32, 1]>>
  %0 = vector.transfer_read %subview[%c0, %c0], %cst {in_bounds = [true, true]} : memref<4x4xbf16, strided<[32, 1]>>, vector<4x4xbf16>
  %1 = arith.extf %0 : vector<4x4xbf16> to vector<4x4xf32>
  vector.print %1 : vector<4x4xf32>
  return %alloc : memref<8x16x32x32xbf16>
}
