// UNSUPPORTED: !x86_64

// RUN: standalone-opt %s -convert-tpp-to-loops -verify-diagnostics

func.func @matmul_to_loops(%arg0: memref<2x8x2xbf16>, %arg1: memref<8x4xbf16>, %arg2: memref<4x4xbf16>) {
 // expected-error @below {{Packed BF16 loops unsupported}}
 // expected-error @below {{'memref.load' op incorrect number of indices for load}}
  tpp.matmul ins(%arg0: memref<2x8x2xbf16>, %arg1: memref<8x4xbf16>)
             out(%arg2: memref<4x4xbf16>)
  return
}

