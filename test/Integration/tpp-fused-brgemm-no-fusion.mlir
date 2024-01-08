// RUN: tpp-run %s \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

memref.global "private" constant @__bias2D : memref<4x4xf32> =
  dense<[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]>

func.func @entry(%arg0: memref<64x4x4xf32>, %arg1: memref<64x4x4xf32>, %arg2: memref<4x4xf32>, %arg3: memref<4x4xf32>) {
  %bias2 = memref.get_global @__bias2D : memref<4x4xf32>
  tpp.fused_brgemm [unary = none, binary = none]
    ins(%arg0 : memref<64x4x4xf32>, %arg1 : memref<64x4x4xf32>,
        %arg3 : memref<4x4xf32>, %bias2 : memref<4x4xf32>)
    outs(%arg3 : memref<4x4xf32>)

  %cast3 = memref.cast %arg3 :memref<4x4xf32> to memref<*xf32>
  call @printMemrefF32(%cast3) : (memref<*xf32>) -> ()

  return
}
func.func private @printMemrefF32(memref<*xf32>)

// CHECK:  [257,   257,   257,   257],
// CHECK:  [257,   257,   257,   257],
// CHECK:  [257,   257,   257,   257],
// CHECK:  [257,   257,   257,   257]
