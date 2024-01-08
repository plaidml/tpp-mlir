// RUN: ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0:${ASAN_OPTIONS} \
// RUN: tpp-run %s -gpu=cuda \
// RUN:  -entry-point-result=void -e entry 2>&1 | \
// RUN: FileCheck %s

func.func @entry() {
  %0, %t0 = gpu.alloc async () : memref<8x8xf32>
  gpu.wait [%t0]
  %1, %t1 = gpu.alloc async () : memref<8x8xf32>
  gpu.wait [%t1]
  %2, %t2 = gpu.alloc async () : memref<8x8xf32>
  gpu.wait [%t2]

  %cst0 = arith.constant 0.0 : f32
  %cst1 = arith.constant 1.0 : f32
  %cst2 = arith.constant 2.0 : f32

  linalg.fill ins(%cst1 : f32) outs(%0 : memref<8x8xf32>)
  linalg.fill ins(%cst2 : f32) outs(%1 : memref<8x8xf32>)
  linalg.fill ins(%cst0 : f32) outs(%2 : memref<8x8xf32>)

  linalg.matmul ins(%0, %1 : memref<8x8xf32>, memref<8x8xf32>)
                outs(%2: memref<8x8xf32>)

  %out = memref.alloc() : memref<8x8xf32>
  %tOut = gpu.memcpy async %out, %2 : memref<8x8xf32>, memref<8x8xf32>
  gpu.wait [%tOut]
  %cast = memref.cast %out : memref<8x8xf32> to memref<*xf32>
  call @printMemrefF32(%cast) : (memref<*xf32>) -> ()

  %tD0 = gpu.dealloc async %0 : memref<8x8xf32>
  gpu.wait [%tD0]
  %tD1 = gpu.dealloc async %1 : memref<8x8xf32>
  gpu.wait [%tD1]
  %tD2 = gpu.dealloc async %2 : memref<8x8xf32>
  gpu.wait [%tD2]

  memref.dealloc %out : memref<8x8xf32>

  return
}

func.func private @printMemrefF32(memref<*xf32>)

// CHECK-COUNT-8: [16,   16,   16,   16,   16,   16,   16,   16]
