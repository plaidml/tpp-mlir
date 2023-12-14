// RUN: tpp-run %s \
// RUN:  -e entry -entry-point-result=void

  memref.global "private" constant @arg1 : memref<128x512x2xbf16> = dense<1.00e+00>
  memref.global "private" constant @arg3 : memref<256x1024x2xbf16> = dense<1.00e+00>
  memref.global "private" constant @arg5 : memref<512x2048x2xbf16> = dense<1.00e+00>
  memref.global "private" constant @arg7 : memref<1024x1000x2xbf16> = dense<1.00e+00>

  func.func @entry(%arg0:memref<128x256xbf16>, %arg2:memref<512xbf16>, %arg4:memref<1024xbf16>, %arg6:memref<2048xbf16>, %arg8:memref<1000xbf16>, %arg9:memref<128x512xbf16>,  %arg10:memref<128x1024xbf16>, %arg11:memref<128x2048xbf16>, %arg12:memref<128x1000xbf16>) {
    tpp.identity ins(%arg2 : memref<512xbf16>) outs(%arg9 : memref<128x512xbf16>)
    %relayout_arg0 = memref.get_global @arg1:memref<128x512x2xbf16>
    tpp.gemm ins(%arg0 : memref<128x256xbf16>, %relayout_arg0 : memref<128x512x2xbf16>, %arg9 : memref<128x512xbf16>) outs(%arg9 : memref<128x512xbf16>)
    tpp.relu ins(%arg9 : memref<128x512xbf16>) outs(%arg9 : memref<128x512xbf16>)

    tpp.identity ins(%arg4 : memref<1024xbf16>) outs(%arg10 : memref<128x1024xbf16>)
    %relayout_arg12 = memref.get_global @arg3:memref<256x1024x2xbf16>
    tpp.gemm ins(%arg9 : memref<128x512xbf16>, %relayout_arg12 : memref<256x1024x2xbf16>, %arg10 : memref<128x1024xbf16>) outs(%arg10 : memref<128x1024xbf16>)
    tpp.relu ins(%arg10 : memref<128x1024xbf16>) outs(%arg10 : memref<128x1024xbf16>)


    tpp.identity ins(%arg6 : memref<2048xbf16>) outs(%arg11 : memref<128x2048xbf16>)
    %relayout_arg11 = memref.get_global @arg5:memref<512x2048x2xbf16>
    tpp.gemm ins(%arg10 : memref<128x1024xbf16>, %relayout_arg11 : memref<512x2048x2xbf16>, %arg11 : memref<128x2048xbf16>) outs(%arg11 : memref<128x2048xbf16>)
    tpp.relu ins(%arg11 : memref<128x2048xbf16>) outs(%arg11 : memref<128x2048xbf16>)

    tpp.identity ins(%arg8 : memref<1000xbf16>) outs(%arg12 : memref<128x1000xbf16>)
    %relayout_arg10 = memref.get_global @arg7:memref<1024x1000x2xbf16>
    tpp.gemm ins(%arg11 : memref<128x2048xbf16>, %relayout_arg10 : memref<1024x1000x2xbf16>, %arg12 : memref<128x1000xbf16>) outs(%arg12 : memref<128x1000xbf16>)
    tpp.relu ins(%arg12 : memref<128x1000xbf16>) outs(%arg12 : memref<128x1000xbf16>)

    %threshold = arith.constant 1.0 : bf16
    %c4 = arith.constant 2.74878e+11: bf16
    %interim4 = memref.alloc(): memref<128x1000xbf16>
    linalg.fill ins(%c4:bf16) outs(%interim4: memref<128x1000xbf16>)
    check.expect_almost_eq(%interim4, %arg12, %threshold): memref<128x1000xbf16>, memref<128x1000xbf16>, bf16

    return
  }
