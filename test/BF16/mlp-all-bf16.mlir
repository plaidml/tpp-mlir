// RUN: tpp-opt %s -linalg-ext-to-loops -convert-check-to-loops -convert-linalg-to-loops -convert-tpp-to-xsmm -convert-xsmm-to-func -convert-vector-to-scf -convert-scf-to-cf -lower-affine -convert-vector-to-llvm -convert-memref-to-llvm -arith-expand -convert-math-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts |\
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlibdir/libmlir_c_runner_utils%shlibext,%tpplibdir/libtpp_c_runner_utils%shlibext
//

module @predict_function {
  func.func @entry(){
    %c0 = arith.constant 1.0:bf16
    %arg0 = memref.alloc():memref<128x256xbf16>
    linalg.fill ins(%c0:bf16) outs(%arg0:memref<128x256xbf16>)
    %arg1 = memref.alloc():memref<256x512xbf16>
    linalg.fill ins(%c0:bf16) outs(%arg1:memref<256x512xbf16>)
    %arg2 = memref.alloc():memref<512xbf16>
    linalg.fill ins(%c0:bf16) outs(%arg2:memref<512xbf16>)
    %arg3 = memref.alloc():memref<512x1024xbf16>
    linalg.fill ins(%c0:bf16) outs(%arg3:memref<512x1024xbf16>)
    %arg4 = memref.alloc(): memref<1024xbf16>
    linalg.fill ins(%c0:bf16) outs(%arg4:memref<1024xbf16>)
    %arg5 = memref.alloc(): memref<1024x2048xbf16>
    linalg.fill ins(%c0:bf16) outs(%arg5:memref<1024x2048xbf16>)
    %arg6 = memref.alloc(): memref<2048xbf16>
    linalg.fill ins(%c0:bf16) outs(%arg6:memref<2048xbf16>)
    %arg7 = memref.alloc(): memref<2048x1000xbf16>
    linalg.fill ins(%c0:bf16) outs(%arg7:memref<2048x1000xbf16>)
    %arg8 = memref.alloc(): memref<1000xbf16>
    linalg.fill ins(%c0:bf16) outs(%arg8:memref<1000xbf16>)
    
    %arg9 = memref.alloc(): memref<128x1000xbf16>
    %arg10 = memref.alloc(): memref<128x2048xbf16>
    %arg11 = memref.alloc(): memref<128x1024xbf16>
    %arg12 = memref.alloc():memref<128x512xbf16>
    
    tpp.identity ins(%arg2 : memref<512xbf16>) out(%arg12 : memref<128x512xbf16>)
    %relayout_arg0 = memref.alloc():memref<128x512x2xbf16>
    linalgx.pack %arg1 inner_dims_pos = [0] inner_tiles = [2] into %relayout_arg0:(memref<256x512xbf16> memref<128x512x2xbf16>)
    tpp.vnni_matmul ins(%arg0 : memref<128x256xbf16>, %relayout_arg0 : memref<128x512x2xbf16>) out(%arg12 : memref<128x512xbf16>)
    tpp.relu out(%arg12 : memref<128x512xbf16>)
    %c1 = arith.constant 256.0: bf16
    %interim1 = memref.alloc(): memref<128x512xbf16>
    linalg.fill ins(%c1:bf16) outs(%interim1: memref<128x512xbf16>)
    %threshold = arith.constant 0.75:bf16
    check.expect_almost_eq(%interim1, %arg12, %threshold): memref<128x512xbf16>, memref<128x512xbf16>, bf16
 
    tpp.identity ins(%arg4 : memref<1024xbf16>) out(%arg11 : memref<128x1024xbf16>)
    %relayout_arg12 = memref.alloc():memref<256x1024x2xbf16>
    linalgx.pack %arg3 inner_dims_pos = [0] inner_tiles = [2] into %relayout_arg12:(memref<512x1024xbf16> memref<256x1024x2xbf16>)
    tpp.vnni_matmul ins(%arg12 : memref<128x512xbf16>, %relayout_arg12 : memref<256x1024x2xbf16>) out(%arg11 : memref<128x1024xbf16>)
    tpp.relu out(%arg11 : memref<128x1024xbf16>)
    %c2 = arith.constant 131360.0: bf16
    %interim2 = memref.alloc(): memref<128x1024xbf16>
    linalg.fill ins(%c2:bf16) outs(%interim2: memref<128x1024xbf16>)
    check.expect_almost_eq(%interim2, %arg11, %threshold): memref<128x1024xbf16>, memref<128x1024xbf16>, bf16

    tpp.identity ins(%arg6 : memref<2048xbf16>) out(%arg10 : memref<128x2048xbf16>)
    %relayout_arg11 = memref.alloc():memref<512x2048x2xbf16>
    linalgx.pack %arg5 inner_dims_pos = [0] inner_tiles = [2] into %relayout_arg11:(memref<1024x2048xbf16> memref<512x2048x2xbf16>)
    tpp.vnni_matmul ins(%arg11 : memref<128x1024xbf16>, %relayout_arg11 : memref<512x2048x2xbf16>) out(%arg10 : memref<128x2048xbf16>)
    tpp.relu out(%arg10 : memref<128x2048xbf16>)
    %c3 = arith.constant 1.34533e+08: bf16
    %interim3 = memref.alloc(): memref<128x2048xbf16>
    linalg.fill ins(%c3:bf16) outs(%interim3: memref<128x2048xbf16>)
    check.expect_almost_eq(%interim3, %arg10, %threshold): memref<128x2048xbf16>, memref<128x2048xbf16>, bf16

    tpp.identity ins(%arg8 : memref<1000xbf16>) out(%arg9 : memref<128x1000xbf16>)
    %relayout_arg10 = memref.alloc():memref<1024x1000x2xbf16>
    linalgx.pack %arg7 inner_dims_pos = [0] inner_tiles = [2] into %relayout_arg10:(memref<2048x1000xbf16> memref<1024x1000x2xbf16>)
    tpp.vnni_matmul ins(%arg10 : memref<128x2048xbf16>, %relayout_arg10 : memref<1024x1000x2xbf16>) out(%arg9 : memref<128x1000xbf16>)
    tpp.relu out(%arg9 : memref<128x1000xbf16>)
    %c4 = arith.constant 2.7557e+11: bf16
    %interim4 = memref.alloc(): memref<128x1000xbf16>
    linalg.fill ins(%c4:bf16) outs(%interim4: memref<128x1000xbf16>)
    check.expect_almost_eq(%interim4, %arg9, %threshold): memref<128x1000xbf16>, memref<128x1000xbf16>, bf16    

    return
  }
}
