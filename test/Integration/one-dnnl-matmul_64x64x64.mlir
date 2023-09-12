// RUN: tpp-run %s \
// RUN:  -e entry -entry-point-result=void -print | \
// RUN: FileCheck %s

func.func private @linalg_matmul_view64x64xf32_view64x64xf32_view64x64xf32(memref<64x64xf32, strided<[?, ?], offset: ?>>, memref<64x64xf32, strided<[?, ?], offset: ?>>, memref<64x64xf32, strided<[?, ?], offset: ?>>) attributes {llvm.emit_c_interface}
  
func.func @entry(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) {
  %cast = memref.cast %arg0 : memref<64x64xf32> to memref<64x64xf32, strided<[?, ?], offset: ?>>
  %cast_0 = memref.cast %arg1 : memref<64x64xf32> to memref<64x64xf32, strided<[?, ?], offset: ?>>
  %cast_1 = memref.cast %arg2 : memref<64x64xf32> to memref<64x64xf32, strided<[?, ?], offset: ?>>
  call @linalg_matmul_view64x64xf32_view64x64xf32_view64x64xf32(%cast, %cast_0, %cast_1) : (memref<64x64xf32, strided<[?, ?], offset: ?>>, memref<64x64xf32, strided<[?, ?], offset: ?>>, memref<64x64xf32, strided<[?, ?], offset: ?>>) -> ()
  return
}

// CHECK-COUNT-64: ( 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65 )
