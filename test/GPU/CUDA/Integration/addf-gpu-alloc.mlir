// RUN: ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0:${ASAN_OPTIONS} \
// RUN: tpp-run %s -gpu=cuda \
// RUN:  -entry-point-result=void -e entry 2>&1 | \
// RUN: FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @kernel_add(%arg0 : memref<8xf32>, %arg1 : memref<8xf32>, %arg2 : memref<8xf32>)
      kernel attributes { gpu.known_block_size = array<i32: 8, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1> } {
      %0 = gpu.block_id x
      %1 = memref.load %arg0[%0] : memref<8xf32>
      %2 = memref.load %arg1[%0] : memref<8xf32>
      %3 = arith.addf %1, %2 : f32
      memref.store %3, %arg2[%0] : memref<8xf32>
      gpu.return
    }
  }

  func.func @entry() {
    %arg0, %t0 = gpu.alloc async () : memref<8xf32>
    gpu.wait [%t0]
    %arg1, %t1 = gpu.alloc async () : memref<8xf32>
    gpu.wait [%t1]
    %arg2, %t2 = gpu.alloc async () : memref<8xf32>
    gpu.wait [%t2]

    %0 = arith.constant 0 : i32
    %1 = arith.constant 1 : i32
    %2 = arith.constant 2 : i32
    %value0 = arith.constant 0.0 : f32
    %value1 = arith.constant 1.1 : f32
    %value2 = arith.constant 2.2 : f32
    linalg.fill ins(%value1 : f32) outs(%arg0 : memref<8xf32>)
    linalg.fill ins(%value2 : f32) outs(%arg1 : memref<8xf32>)
    linalg.fill ins(%value0 : f32) outs(%arg2 : memref<8xf32>)

    %cst1 = arith.constant 1 : index
    %cst8 = arith.constant 8 : index
    gpu.launch_func @kernels::@kernel_add
        blocks in (%cst8, %cst1, %cst1) threads in (%cst1, %cst1, %cst1)
        args(%arg0 : memref<8xf32>, %arg1 : memref<8xf32>, %arg2 : memref<8xf32>)

    %out = memref.alloc() : memref<8xf32>
    %tOut = gpu.memcpy async %out, %arg2 : memref<8xf32>, memref<8xf32>
    gpu.wait [%tOut]
    %cast = memref.cast %out : memref<8xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()

    %tD0 = gpu.dealloc async %arg0 : memref<8xf32>
    gpu.wait [%tD0]
    %tD1 = gpu.dealloc async %arg1 : memref<8xf32>
    gpu.wait [%tD1]
    %tD2 = gpu.dealloc async %arg2 : memref<8xf32>
    gpu.wait [%tD2]

    memref.dealloc %out : memref<8xf32>

    return
  }
  func.func private @printMemrefF32(%ptr : memref<*xf32>)
}

// CHECK: [3.3,  3.3,  3.3,  3.3,  3.3,  3.3,  3.3,  3.3]
