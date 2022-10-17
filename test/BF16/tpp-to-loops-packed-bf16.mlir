// RUN: tpp-opt %s -convert-tpp-to-loops -verify-diagnostics

func.func @matmul_to_loops(%arg0: memref<2x8x2xbf16>, %arg1: memref<8x4xbf16>, %arg2: memref<4x4xbf16>) {
  tpp.matmul ins(%arg0: memref<2x8x2xbf16>, %arg1: memref<8x4xbf16>)
             out(%arg2: memref<4x4xbf16>)
  return
}
