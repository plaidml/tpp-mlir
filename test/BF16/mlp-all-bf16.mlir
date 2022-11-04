// RUN: tpp-opt %s -linalg-ext-to-loops -convert-linalg-to-loops -convert-tpp-to-xsmm -convert-xsmm-to-func -convert-vector-to-scf -convert-scf-to-cf -lower-affine -convert-vector-to-llvm -convert-memref-to-llvm -arith-expand -convert-math-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts |\
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlirdir/libmlir_c_runner_utils%shlibext,%tpplibdir/libtpp_c_runner_utils%shlibext
//

module @predict_function {
  func.func @entry(){
    %arg0 = memref.alloc():memref<128x256xbf16>
    %arg1 = memref.alloc():memref<256x512xbf16>
    %arg2 = memref.alloc():memref<512xbf16>
    %arg3 = memref.alloc():memref<512x1024xbf16>
    %arg4 = memref.alloc(): memref<1024xbf16>
    %arg5 = memref.alloc(): memref<1024x2048xbf16>
    %arg6 = memref.alloc(): memref<2048xbf16>
    %arg7 = memref.alloc(): memref<2048x1000xbf16>
    %arg8 = memref.alloc(): memref<1000xbf16>
    %arg9 = memref.alloc(): memref<128x1000xbf16>
    %arg10 = memref.alloc(): memref<128x2048xbf16>
    %arg11 = memref.alloc(): memref<128x1024xbf16>
    %arg12 = memref.alloc():memref<128x512xbf16>
    tpp.identity ins(%arg2 : memref<512xbf16>) out(%arg12 : memref<128x512xbf16>)
    %relayout_arg0 = memref.alloc():memref<64x256x2xbf16>
    linalgx.pack %arg0 inner_dims_pos = [0] inner_tiles = [2] into %relayout_arg0:(memref<128x256xbf16> memref<64x256x2xbf16>)
    tpp.matmul ins(%relayout_arg0 : memref<64x256x2xbf16>, %arg1 : memref<256x512xbf16>) out(%arg12 : memref<128x512xbf16>)
    tpp.relu ins(%arg12 : memref<128x512xbf16>) out(%arg12 : memref<128x512xbf16>)
    tpp.identity ins(%arg4 : memref<1024xbf16>) out(%arg11 : memref<128x1024xbf16>)
    %relayout_arg12 = memref.alloc():memref<64x512x2xbf16>
    linalgx.pack %arg12 inner_dims_pos = [0] inner_tiles = [2] into %relayout_arg12:(memref<128x512xbf16> memref<64x512x2xbf16>)
    tpp.matmul ins(%relayout_arg12 : memref<64x512x2xbf16>, %arg3 : memref<512x1024xbf16>) out(%arg11 : memref<128x1024xbf16>)
    tpp.relu ins(%arg11 : memref<128x1024xbf16>) out(%arg11 : memref<128x1024xbf16>)
    tpp.identity ins(%arg6 : memref<2048xbf16>) out(%arg10 : memref<128x2048xbf16>)
    %relayout_arg11 = memref.alloc():memref<64x1024x2xbf16>
    linalgx.pack %arg11 inner_dims_pos = [0] inner_tiles = [2] into %relayout_arg11:(memref<128x1024xbf16> memref<64x1024x2xbf16>)
    tpp.matmul ins(%relayout_arg11 : memref<64x1024x2xbf16>, %arg5 : memref<1024x2048xbf16>) out(%arg10 : memref<128x2048xbf16>)
    tpp.relu ins(%arg10 : memref<128x2048xbf16>) out(%arg10 : memref<128x2048xbf16>)
    tpp.identity ins(%arg8 : memref<1000xbf16>) out(%arg9 : memref<128x1000xbf16>)
    %relayout_arg10 = memref.alloc():memref<64x2048x2xbf16>
    linalgx.pack %arg10 inner_dims_pos = [0] inner_tiles = [2] into %relayout_arg10:(memref<128x2048xbf16> memref<64x2048x2xbf16>)
    tpp.matmul ins(%relayout_arg10 : memref<64x2048x2xbf16>, %arg7 : memref<2048x1000xbf16>) out(%arg9 : memref<128x1000xbf16>)
    tpp.relu ins(%arg9 : memref<128x1000xbf16>) out(%arg9 : memref<128x1000xbf16>)
    return
  }
}
