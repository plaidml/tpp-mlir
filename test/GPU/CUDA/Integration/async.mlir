// RUN: ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0:${ASAN_OPTIONS} \
// RUN: tpp-run %s -gpu=cuda \
// RUN:  -entry-point-result=void -e entry 2>&1 | \
// RUN: FileCheck %s

// Original test from: llvm-project/mlir/test/Integration/GPU/CUDA/async.mlir

func.func @entry() {
  %c0    = arith.constant 0 : index
  %c1    = arith.constant 1 : index
  %count = arith.constant 2 : index

  // initialize h0 on host
  %h0 = memref.alloc(%count) : memref<?xi32>
  %h0_unranked = memref.cast %h0 : memref<?xi32> to memref<*xi32>
  gpu.host_register %h0_unranked : memref<*xi32>

  %v0 = arith.constant 42 : i32
  memref.store %v0, %h0[%c0] : memref<?xi32>
  memref.store %v0, %h0[%c1] : memref<?xi32>

  // copy h0 to b0 on device.
  %t0, %f0 = async.execute () -> !async.value<memref<?xi32>> {
    %b0 = gpu.alloc(%count) : memref<?xi32>
    gpu.memcpy %b0, %h0 : memref<?xi32>, memref<?xi32>
    async.yield %b0 : memref<?xi32>
  }

  // copy h0 to b1 and b2 (fork)
  %t1, %f1 = async.execute [%t0] (
    %f0 as %b0 : !async.value<memref<?xi32>>
  ) -> !async.value<memref<?xi32>> {
    %b1 = gpu.alloc(%count) : memref<?xi32>
    gpu.memcpy %b1, %b0 : memref<?xi32>, memref<?xi32>
    async.yield %b1 : memref<?xi32>
  }
  %t2, %f2 = async.execute [%t0] (
    %f0 as %b0 : !async.value<memref<?xi32>>
  ) -> !async.value<memref<?xi32>> {
    %b2 = gpu.alloc(%count) : memref<?xi32>
    gpu.memcpy %b2, %b0 : memref<?xi32>, memref<?xi32>
    async.yield %b2 : memref<?xi32>
  }

  // h0 = b1 + b2 (join).
  %t3 = async.execute [%t1, %t2] (
    %f1 as %b1 : !async.value<memref<?xi32>>,
    %f2 as %b2 : !async.value<memref<?xi32>>
  ) {
    gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
               threads(%tx, %ty, %tz) in (%block_x = %count, %block_y = %c1, %block_z = %c1) {
      %v1 = memref.load %b1[%tx] : memref<?xi32>
      %v2 = memref.load %b2[%tx] : memref<?xi32>
      %sum = arith.addi %v1, %v2 : i32
      memref.store %sum, %h0[%tx] : memref<?xi32>
      gpu.terminator
    }
    async.yield
  }

  async.await %t3 : !async.token
  call @printMemrefI32(%h0_unranked) : (memref<*xi32>) -> ()
  return
}

func.func private @printMemrefI32(memref<*xi32>)

// TODO check real values when 'CUDA_ERROR_ILLEGAL_ADDRESS' bug is resolved
// [84, 84]
// CHECK: {{\[}}{{-?}}{{[0-9]+}}, {{-?}}{{[0-9]+}}
