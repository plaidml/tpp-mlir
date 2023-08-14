// RUN: ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0:${ASAN_OPTIONS} \
// RUN: tpp-run %s -gpu=vulkan \
// RUN:  -entry-point-result=void -e entry 2>&1 | \
// RUN: FileCheck %s

func.func @entry() {
  %0 = memref.alloc() : memref<8x8xf32>
  %1 = memref.alloc() : memref<8x8xf32>
  %2 = memref.alloc() : memref<8x8xf32>

  %cst0 = arith.constant 0.0 : f32
  %cst1 = arith.constant 1.0 : f32
  %cst2 = arith.constant 2.0 : f32

  %cast0 = memref.cast %0 : memref<8x8xf32> to memref<?x?xf32>
  %cast1 = memref.cast %1 : memref<8x8xf32> to memref<?x?xf32>
  %cast2 = memref.cast %2 : memref<8x8xf32> to memref<?x?xf32>
  call @fillResource2DFloat(%cast0, %cst1) : (memref<?x?xf32>, f32) -> ()
  call @fillResource2DFloat(%cast1, %cst2) : (memref<?x?xf32>, f32) -> ()
  call @fillResource2DFloat(%cast2, %cst0) : (memref<?x?xf32>, f32) -> ()

  tpp.gemm ins(%0 : memref<8x8xf32>, %1 : memref<8x8xf32>, %2: memref<8x8xf32>)
           outs(%2: memref<8x8xf32>)

  %castOut = memref.cast %cast2 : memref<?x?xf32> to memref<*xf32>
  call @printMemrefF32(%castOut) : (memref<*xf32>) -> ()

  memref.dealloc %0 : memref<8x8xf32>
  memref.dealloc %1 : memref<8x8xf32>
  memref.dealloc %2 : memref<8x8xf32>

  return
}
func.func private @fillResource2DFloat(%0 : memref<?x?xf32>, %1 : f32)
func.func private @printMemrefF32(memref<*xf32>)

// CHECK-COUNT-8: {{\[}}16,   16,   16,   16,   16,   16,   16,   16{{\]}}
