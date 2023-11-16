// RUN: tpp-opt %s -verify-xsmm-calls -verify-diagnostics -split-input-file
// Make sure we do not emit any error here.

func.func @identity(%arg0: memref<1x1xf32, strided<[8, 1], offset: ?>>, 
                    %arg1: memref<1x1xf32, strided<[8, 1], offset: ?>>) {
  %0 = xsmm.unary.dispatch identity [1, 1, 8, 8] flags = (none) data_type = f32
  xsmm.unary identity(data_type = f32, %0, %arg0, %arg1) 
    : (i64, memref<1x1xf32, strided<[8, 1], offset: ?>>, memref<1x1xf32, strided<[8, 1], offset: ?>>) -> ()
  return
}

// -----

func.func @identity(%arg0: f32, 
                    %arg1: memref<1x1xf32, strided<[8, 1], offset: ?>>) {
  %0 = xsmm.unary.dispatch identity [1, 1, 8, 8] flags = (bcast_scalar) data_type = f32
  xsmm.unary identity(data_type = f32, %0, %arg0, %arg1) 
    : (i64, f32, memref<1x1xf32, strided<[8, 1], offset: ?>>) -> ()
  return
}

// -----

func.func @identity(%arg0: f32, %arg1: memref<1x1xf32>) {
  %0 = xsmm.unary.dispatch identity [1, 1, 1, 1] flags = (bcast_scalar) data_type = f32
  xsmm.unary identity(data_type = f32, %0, %arg0, %arg1) 
    : (i64, f32, memref<1x1xf32>) -> ()
  return
}

// -----

func.func @gemm(%arg0: memref<3x6x2xbf16>, %arg1: memref<6x6xbf16>) {
  %0 = xsmm.gemm.dispatch [6, 6, 6, 6, 6, 6] flags = (vnni_a) data_type = bf16
  xsmm.gemm(data_type = bf16, %0, %arg0, %arg0, %arg1) :
    (i64, memref<3x6x2xbf16>, memref<3x6x2xbf16>, memref<6x6xbf16>) -> ()
  return
}
