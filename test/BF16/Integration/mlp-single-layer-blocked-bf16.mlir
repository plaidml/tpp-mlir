// RUN:tpp-opt %s -convert-linalg-to-loops -convert-tpp-to-xsmm -convert-xsmm-to-func -convert-vector-to-scf -convert-scf-to-cf -expand-strided-metadata -lower-affine -convert-arith-to-llvm -convert-vector-to-llvm -finalize-memref-to-llvm -arith-expand -convert-math-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts |\
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlibdir/libmlir_c_runner_utils%shlibext,%tpplibdir/libtpp_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//

// Validate default pipeline
// RUN: tpp-run %s -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

func.func @entry(){
  %f0 = arith.constant 1.0:bf16
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  %alloc = memref.alloc() {alignment = 128 : i64} : memref<32x64x4x4xbf16>
  linalg.fill ins(%f0:bf16) outs(%alloc:memref<32x64x4x4xbf16>)
  %wt_alloc = memref.alloc() {alignment = 128 : i64} : memref<128x64x2x4x2xbf16>
  linalg.fill ins(%f0:bf16) outs(%wt_alloc:memref<128x64x2x4x2xbf16>)
  %alloc_1 = memref.alloc() {alignment = 128 : i64} : memref<32x128x4x4xbf16>
  linalg.fill ins(%f0:bf16) outs(%alloc_1:memref<32x128x4x4xbf16>)
  scf.for %arg4 = %c0 to %c4 step %c1 {
    %subview = memref.subview %alloc[%arg4, 0, 0, 0] [1, 64, 4, 4] [1, 1, 1, 1] : memref<32x64x4x4xbf16> to memref<64x4x4xbf16, strided<[16, 4, 1], offset: ?>>
    %subview_2 = memref.subview %alloc_1[%arg4, 0, 0, 0] [1, 128, 4, 4] [1, 1, 1, 1] : memref<32x128x4x4xbf16> to memref<128x4x4xbf16, strided<[16, 4, 1], offset: ?>>
    scf.for %arg5 = %c0 to %c16 step %c1 {
      %subview_3 = memref.subview %wt_alloc[%arg5, 0, 0, 0, 0] [1, 64, 2, 4, 2] [1, 1, 1, 1, 1] : memref<128x64x2x4x2xbf16> to memref<64x2x4x2xbf16, strided<[16, 8, 2, 1], offset: ?>>
      %subview_4 = memref.subview %subview_2[%arg5, 0, 0] [1, 4, 4] [1, 1, 1] : memref<128x4x4xbf16, strided<[16, 4, 1], offset: ?>> to memref<4x4xbf16, strided<[4, 1], offset: ?>>
      tpp.brgemm ins(%subview:memref<64x4x4xbf16, strided<[16, 4,1], offset:?>>, %subview_3 : memref<64x2x4x2xbf16, strided<[16, 8, 2, 1], offset: ?>>, %subview_4 : memref<4x4xbf16, strided<[4, 1], offset: ?>>) outs(%subview_4 : memref<4x4xbf16, strided<[4, 1], offset: ?>>)
      tpp.relu ins(%subview_4 : memref<4x4xbf16, strided<[4, 1], offset: ?>>) outs(%subview_4 : memref<4x4xbf16, strided<[4, 1], offset: ?>>)
      %d1 = arith.constant -1.0 : bf16
      %v0 = vector.transfer_read %subview_4[%c0, %c0], %d1 : memref<4x4xbf16, strided<[4,1], offset:?>>, vector<4x4xbf16>
      %f1 = arith.extf %v0:vector<4x4xbf16> to vector<4x4xf32>

      //
      // CHECK:( ( 256, 256, 256, 256 ), ( 256, 256, 256, 256 ), ( 256, 256, 256, 256 ), ( 256, 256, 256, 256 ) )
      //
      vector.print %f1 : vector<4x4xf32>
    }
  }
  memref.dealloc %alloc : memref<32x64x4x4xbf16>
  memref.dealloc %wt_alloc : memref<128x64x2x4x2xbf16>
  memref.dealloc %alloc_1 : memref<32x128x4x4xbf16>
  return
}
