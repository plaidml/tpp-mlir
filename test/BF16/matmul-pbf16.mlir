// RUN: tpp-opt %s -linalg-ext-to-loops -convert-linalg-to-loops -convert-tpp-to-xsmm -convert-xsmm-to-func -convert-vector-to-scf -convert-scf-to-cf -sparse-compiler |\
// RUN: tpp-run \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlirdir/libmlir_c_runner_utils%shlibext,%tpplibdir/libtpp_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//

module {
 func.func @matmultpp(%A: memref<2x8x2xbf16>, 
          %B: memref<8x4xbf16>, %C: memref<4x4xbf16>) attributes {llvm.emit_c_interface} {
    tpp.matmul ins(%A: memref<2x8x2xbf16>, %B: memref<8x4xbf16>)
             out(%C: memref<4x4xbf16>)
    return
  }

  func.func @entry() {
    %c0 = arith.constant 0 : index
    %f0 = arith.constant 1.0 : bf16
    %da = memref.alloc() :memref<4x8xbf16>
    linalg.fill ins(%f0 : bf16) outs(%da : memref<4x8xbf16>)
    %db = memref.alloc() :memref<8x4xbf16>
    linalg.fill ins(%f0:bf16) outs (%db:memref<8x4xbf16>)
    // Call kernel.
    %0 = memref.alloc() : memref<2x8x2xbf16>
    linalgx.pack %da inner_dims_pos = [0] inner_tiles = [2] into %0 : (memref<4x8xbf16> memref<2x8x2xbf16>)    
    %D = memref.alloc() : memref<4x4xbf16>
    %zero = arith.constant 0.0 : bf16
    linalg.fill ins(%zero : bf16) outs(%D:memref<4x4xbf16>)
    call @matmultpp(%0, %db, %D)
       : (memref<2x8x2xbf16>, memref<8x4xbf16>, memref<4x4xbf16>)->()

    //
    // CHECK:( ( 8, 8, 8, 8 ), ( 8, 8, 8, 8 ), ( 8, 8, 8, 8 ), ( 8, 8, 8, 8 ) )
    //
    %d1 = arith.constant -1.0 : bf16

    %v0 = vector.transfer_read %D[%c0, %c0], %d1 : memref<4x4xbf16>, vector<4x4xbf16>
    %f1 = arith.extf %v0:vector<4x4xbf16> to vector<4x4xf32>
    vector.print %f1 : vector<4x4xf32>

    return
  }
 
}
