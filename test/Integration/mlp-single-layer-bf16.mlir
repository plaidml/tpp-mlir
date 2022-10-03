// UNSUPPORTED: !x86_64

// RUN: standalone-opt %s -convert-linalg-to-loops -convert-tpp-to-xsmm -convert-xsmm-to-func -convert-vector-to-scf -convert-scf-to-cf -convert-vector-to-llvm -convert-func-to-llvm -convert-memref-to-llvm -canonicalize -sparse-compiler|\
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlirdir/libmlir_c_runner_utils%shlibext,%standalonelibdir/libstandalone_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//

#map0 = affine_map<(d0,d1,d2)->(d0*2+d2, d1)>
#map1 = affine_map<(d0,d1,d2)->(d0,d1,d2)>
module{
  func.func @entry(){
    %c0 = arith.constant 1.0:bf16
    %arg0 = memref.alloc():memref<128x256xbf16>
    linalg.fill ins(%c0:bf16) outs(%arg0:memref<128x256xbf16>)
    %arg1 = memref.alloc():memref<256x512xbf16>
    linalg.fill ins(%c0:bf16) outs(%arg1:memref<256x512xbf16>)
    %arg2 = memref.alloc():memref<512xbf16>
    linalg.fill ins(%c0:bf16) outs(%arg2:memref<512xbf16>)
    %arg3 = memref.alloc():memref<128x512xbf16>
    linalg.fill ins(%c0:bf16) outs(%arg3:memref<128x512xbf16>)
    tpp.identity ins(%arg2 : memref<512xbf16>) out(%arg3 : memref<128x512xbf16>)
    %wt = memref.alloc():memref<64x256x2xbf16>
    linalgx.relayout ins(%arg0:memref<128x256xbf16>, #map0) outs(%wt:memref<64x256x2xbf16>, #map1)
    tpp.matmul ins(%wt : memref<64x256x2xbf16>, %arg1 : memref<256x512xbf16>) out(%arg3 : memref<128x512xbf16>)
    tpp.relu ins(%arg3 : memref<128x512xbf16>) out(%arg3 : memref<128x512xbf16>)
    return
  }
}

