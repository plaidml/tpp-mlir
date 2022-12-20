// RUN: tpp-opt %s -convert-check-to-loops -linalg-ext-to-loops -convert-linalg-to-loops -convert-tpp-to-xsmm -convert-xsmm-to-func -arith-expand -convert-math-to-llvm  -convert-vector-to-scf -convert-scf-to-cf -lower-affine  -reconcile-unrealized-casts |\
// RUN: tpp-run \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlibdir/libmlir_c_runner_utils%shlibext,%tpplibdir/libtpp_c_runner_utils%shlibext
//

  memref.global "private" constant @arg1 : memref<256x512xbf16> = dense<1.00e+00> 
  memref.global "private" constant @arg3 : memref<512x1024xbf16> = dense<1.00e+00>
  memref.global "private" constant @arg5 : memref<1024x2048xbf16> = dense<1.00e+00>
  memref.global "private" constant @arg7 : memref<2048x1000xbf16> = dense<1.00e+00>

  func.func @entry(%arg0:memref<128x256xbf16>, %arg2:memref<512xbf16>, %arg4:memref<1024xbf16>, %arg6:memref<2048xbf16>, %arg8:memref<1000xbf16>, %arg9:memref<128x512xbf16>,  %arg10:memref<128x1024xbf16>, %arg11:memref<128x2048xbf16>, %arg12:memref<128x1000xbf16>) {
    tpp.identity ins(%arg2 : memref<512xbf16>) out(%arg9 : memref<128x512xbf16>)
    %relayout_arg0 = memref.alloc():memref<128x512x2xbf16>
    %0 = memref.get_global @arg1:memref<256x512xbf16>
    linalgx.pack %0 inner_dims_pos = [0] inner_tiles = [2] into %relayout_arg0:(memref<256x512xbf16> memref<128x512x2xbf16>)
    tpp.vnni_matmul ins(%arg0 : memref<128x256xbf16>, %relayout_arg0 : memref<128x512x2xbf16>) out(%arg9 : memref<128x512xbf16>)
    tpp.relu out(%arg9 : memref<128x512xbf16>)
 
    tpp.identity ins(%arg4 : memref<1024xbf16>) out(%arg10 : memref<128x1024xbf16>)
    %relayout_arg12 = memref.alloc():memref<256x1024x2xbf16>
    %1 = memref.get_global @arg3:memref<512x1024xbf16>
    linalgx.pack %1 inner_dims_pos = [0] inner_tiles = [2] into %relayout_arg12:(memref<512x1024xbf16> memref<256x1024x2xbf16>)
    tpp.vnni_matmul ins(%arg9 : memref<128x512xbf16>, %relayout_arg12 : memref<256x1024x2xbf16>) out(%arg10 : memref<128x1024xbf16>)
    tpp.relu out(%arg10 : memref<128x1024xbf16>)

    tpp.identity ins(%arg6 : memref<2048xbf16>) out(%arg11 : memref<128x2048xbf16>)
    %relayout_arg11 = memref.alloc():memref<512x2048x2xbf16>
    %2 = memref.get_global @arg5:memref<1024x2048xbf16>
    linalgx.pack %2 inner_dims_pos = [0] inner_tiles = [2] into %relayout_arg11:(memref<1024x2048xbf16> memref<512x2048x2xbf16>)
    tpp.vnni_matmul ins(%arg10 : memref<128x1024xbf16>, %relayout_arg11 : memref<512x2048x2xbf16>) out(%arg11 : memref<128x2048xbf16>)
    tpp.relu out(%arg11 : memref<128x2048xbf16>)

    tpp.identity ins(%arg8 : memref<1000xbf16>) out(%arg12 : memref<128x1000xbf16>)
    %relayout_arg10 = memref.alloc():memref<1024x1000x2xbf16>
    %3 = memref.get_global @arg7:memref<2048x1000xbf16>
    linalgx.pack %3 inner_dims_pos = [0] inner_tiles = [2] into %relayout_arg10:(memref<2048x1000xbf16> memref<1024x1000x2xbf16>)
    tpp.vnni_matmul ins(%arg11 : memref<128x2048xbf16>, %relayout_arg10 : memref<1024x1000x2xbf16>) out(%arg12 : memref<128x1000xbf16>)
    tpp.relu out(%arg12 : memref<128x1000xbf16>)

    %threshold = arith.constant 1.0 : bf16
    %c4 = arith.constant 2.74878e+11: bf16
    %interim4 = memref.alloc(): memref<128x1000xbf16>
    linalg.fill ins(%c4:bf16) outs(%interim4: memref<128x1000xbf16>)
    check.expect_almost_eq(%interim4, %arg12, %threshold): memref<128x1000xbf16>, memref<128x1000xbf16>, bf16
    memref.dealloc %interim4: memref<128x1000xbf16>

    return
  }
