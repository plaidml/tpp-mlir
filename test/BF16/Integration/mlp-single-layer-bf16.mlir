// Validate default pipeline
// RUN: tpp-run %s \
// RUN:  -e entry -entry-point-result=void

func.func @entry(){
  %c0 = arith.constant 1.0:bf16
  %arg0 = memref.alloc():memref<128x256xbf16>
  linalg.fill ins(%c0:bf16) outs(%arg0:memref<128x256xbf16>)
  %arg2 = memref.alloc():memref<512xbf16>
  linalg.fill ins(%c0:bf16) outs(%arg2:memref<512xbf16>)
  %arg3 = memref.alloc():memref<128x512xbf16>
  linalg.fill ins(%c0:bf16) outs(%arg3:memref<128x512xbf16>)
  tpp.identity ins(%arg2 : memref<512xbf16>) outs(%arg3 : memref<128x512xbf16>)
  %wt = memref.alloc():memref<128x512x2xbf16>
  linalg.fill ins(%c0:bf16) outs(%wt:memref<128x512x2xbf16>)
  tpp.gemm ins(%arg0 : memref<128x256xbf16>, %wt : memref<128x512x2xbf16>, %arg3 : memref<128x512xbf16>) outs(%arg3 : memref<128x512xbf16>)
  tpp.relu ins(%arg3 : memref<128x512xbf16>) outs(%arg3 : memref<128x512xbf16>)
  %result = memref.alloc():memref<128x512xbf16>
  %c1 = arith.constant 256.0:bf16
  linalg.fill ins(%c1:bf16) outs(%result:memref<128x512xbf16>)
  %threshold = arith.constant 1.0:bf16
  check.expect_almost_eq(%arg3, %result, %threshold): memref<128x512xbf16>, memref<128x512xbf16>, bf16
  // Explicit manual buffer deallocation is required as BufferDeallocationPass
  // inserts automatic deallocation before type casts used in generated XSMM
  // function calls which invalidates the memory before computation.
  memref.dealloc %arg0 : memref<128x256xbf16>
  memref.dealloc %arg2 : memref<512xbf16>
  memref.dealloc %arg3 : memref<128x512xbf16>
  memref.dealloc %wt : memref<128x512x2xbf16>
  memref.dealloc %result : memref<128x512xbf16>
  return
}
