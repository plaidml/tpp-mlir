// RUN:tpp-opt %s -linalg-ext-to-loops -convert-linalg-to-loops -convert-tpp-to-xsmm -convert-xsmm-to-func -convert-vector-to-scf -convert-scf-to-cf -lower-affine -convert-vector-to-llvm -convert-memref-to-llvm -arith-expand -convert-math-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts |\
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlirdir/libmlir_c_runner_utils%shlibext,%tpplibdir/libtpp_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//

func.func @entry(){
  %arg0 = memref.alloc(): memref<128x256xbf16>
  %arg1 = memref.alloc(): memref<256x512xbf16>
  %arg2 = memref.alloc(): memref<512xbf16> 
  %arg3 = memref.alloc(): memref<128x512xbf16>
  %f0 = arith.constant 1.0:bf16
  linalg.fill ins(%f0:bf16) outs(%arg0:memref<128x256xbf16>)
  linalg.fill ins(%f0:bf16) outs(%arg1:memref<256x512xbf16>)
  linalg.fill ins(%f0:bf16) outs(%arg2:memref<512xbf16>)
  linalg.fill ins(%f0:bf16) outs(%arg3:memref<128x512xbf16>)
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  tpp.identity ins(%arg2 : memref<512xbf16>) out(%arg3 : memref<128x512xbf16>)
  %alloc = memref.alloc() {alignment = 128 : i64} : memref<32x64x4x4xbf16>
  %wt_alloc = memref.alloc() {alignment = 128 : i64} : memref<32x32x4x4x2xbf16>
  linalgx.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %alloc : (memref<128x256xbf16> memref<32x64x4x4xbf16>)
  linalgx.pack %alloc inner_dims_pos = [1] inner_tiles = [2] into %wt_alloc : (memref<32x64x4x4xbf16> memref<32x32x4x4x2xbf16>)
  %alloc_0 = memref.alloc() {alignment = 128 : i64} : memref<128x64x4x4xbf16>
  linalgx.pack %arg1 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %alloc_0 : (memref<256x512xbf16> memref<128x64x4x4xbf16>)
  %alloc_1 = memref.alloc() {alignment = 128 : i64} : memref<32x128x4x4xbf16>
  linalgx.pack %arg3 inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %alloc_1 : (memref<128x512xbf16> memref<32x128x4x4xbf16>)
  scf.for %arg4 = %c0 to %c4 step %c1 {
    %subview = memref.subview %wt_alloc[%arg4, 0, 0, 0, 0] [1, 32, 4, 4, 2] [1, 1, 1, 1, 1] : memref<32x32x4x4x2xbf16> to memref<32x4x4x2xbf16, strided<[32, 8, 2, 1], offset: ?>>
    %subview_2 = memref.subview %alloc_1[%arg4, 0, 0, 0] [1, 128, 4, 4] [1, 1, 1, 1] : memref<32x128x4x4xbf16> to memref<128x4x4xbf16, strided<[16, 4, 1], offset: ?>>
    scf.for %arg5 = %c0 to %c16 step %c1 {
      %subview_3 = memref.subview %alloc_0[%arg5, 0, 0, 0] [1, 64, 4, 4] [1, 1, 1, 1] : memref<128x64x4x4xbf16> to memref<64x4x4xbf16, strided<[16, 4, 1], offset: ?>>
      %subview_4 = memref.subview %subview_2[%arg5, 0, 0] [1, 4, 4] [1, 1, 1] : memref<128x4x4xbf16, strided<[16, 4, 1], offset: ?>> to memref<4x4xbf16, strided<[4, 1], offset: ?>>
      tpp.vnni_brgemm ins(%subview:memref<32x4x4x2xbf16, strided<[32,8,2,1], offset:?>>, %subview_3 : memref<64x4x4xbf16, strided<[16, 4, 1], offset: ?>>) out(%subview_4 : memref<4x4xbf16, strided<[4, 1], offset: ?>>)
      tpp.relu out(%subview_4 : memref<4x4xbf16, strided<[4, 1], offset: ?>>)
      %d1 = arith.constant -1.0 : bf16
      %v0 = vector.transfer_read %subview_4[%c0, %c0], %d1 : memref<4x4xbf16, strided<[4,1], offset:?>>, vector<4x4xbf16>
      %f1 = arith.extf %v0:vector<4x4xbf16> to vector<4x4xf32>
        
      //
      // CHECK:( ( 256, 256, 256, 256 ), ( 256, 256, 256, 256 ), ( 256, 256, 256, 256 ), ( 256, 256, 256, 256 ) )
      //
      vector.print %f1 : vector<4x4xf32>
    }
  }
  linalgx.unpack %alloc_1 inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %arg3 : (memref<32x128x4x4xbf16> memref<128x512xbf16>)
  memref.dealloc %alloc : memref<32x64x4x4xbf16>
  memref.dealloc %wt_alloc : memref<32x32x4x4x2xbf16>
  memref.dealloc %alloc_0 : memref<128x64x4x4xbf16>
  memref.dealloc %alloc_1 : memref<32x128x4x4xbf16>
  memref.dealloc %arg0 : memref<128x256xbf16>
  memref.dealloc %arg1 : memref<256x512xbf16>
  memref.dealloc %arg2 : memref<512xbf16>
  memref.dealloc %arg3 : memref<128x512xbf16>
  return
}
