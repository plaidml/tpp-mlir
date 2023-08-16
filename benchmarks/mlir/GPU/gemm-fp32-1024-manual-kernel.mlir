// RUN: tpp-run %s -n 10 \
// RUN:  -e entry -entry-point-result=void

// BENCH_TOTAL_FLOPS: 536870912

module attributes {gpu.container_module} {
  func.func @entry(%arg0: memref<256x1024xf32>, %arg1: memref<1024x1024xf32>, %arg2: memref<256x1024xf32>) {
    %blocksX = arith.constant 8 : index
    %blocksY = arith.constant 32 : index
    %threads = arith.constant 32 : index
    %m = arith.constant 256 : index
    %n = arith.constant 1024 : index
    %k = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    gpu.launch_func  @entry_kernel::@entry_kernel
      blocks in (%blocksX, %blocksY, %c1)
      threads in (%threads, %threads, %c1)
      args(%arg0 : memref<256x1024xf32>, %arg1 : memref<1024x1024xf32>, %arg2 : memref<256x1024xf32>, %m : index, %n : index, %k : index)
    return
  }

  gpu.module @entry_kernel {
    gpu.func @entry_kernel(%arg0: memref<256x1024xf32>, %arg1: memref<1024x1024xf32>, %arg2: memref<256x1024xf32>, %m: index, %n: index, %k: index)
    kernel attributes {gpu.known_block_size = array<i32: 32, 32, 1>, gpu.known_grid_size = array<i32: 8, 32, 1>} {
      %bx = gpu.block_id x
      %by = gpu.block_id y
      %bDimx = gpu.block_dim x
      %bDimy = gpu.block_dim y
      %tx = gpu.thread_id x
      %ty = gpu.thread_id y

      // row = blockIdx.x * blockDim.x + threadIdx.x
      %rowOff = arith.muli %bx, %bDimx : index
      %row = arith.addi %rowOff, %tx : index

      // col = blockIdx.y * blockDim.y + threadIdx.y
      %colOff = arith.muli %by, %bDimy : index
      %col = arith.addi %colOff, %ty : index

      %rowCheck = arith.cmpi ult, %row, %m : index
      %colCheck = arith.cmpi ult, %col, %n : index
      %isValidThread = arith.andi %rowCheck, %colCheck : i1

      scf.if %isValidThread {
        %lb = arith.constant 0 : index
        %step = arith.constant 1 : index
        %init = memref.load %arg2[%row, %col] : memref<256x1024xf32>

        %sum = scf.for %i = %lb to %k step %step iter_args(%partial = %init) -> (f32) {
          %2 = memref.load %arg0[%row, %i] : memref<256x1024xf32>
          %3 = memref.load %arg1[%i, %col] : memref<1024x1024xf32>
          %5 = arith.mulf %2, %3 : f32
          %6 = arith.addf %partial, %5 : f32
          scf.yield %6 : f32
        }

        memref.store %sum, %arg2[%row, %col] : memref<256x1024xf32>
      }

      gpu.return
    }
  }
}
