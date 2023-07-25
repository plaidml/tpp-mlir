// RUN: tpp-opt %s -gpu-flat-args -split-input-file | \
// RUN: FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @gpu_modules {
    gpu.func @non_empty_kernel(%arg0: memref<32x32xf32>)
    kernel attributes {gpu.known_block_size = array<i32: 4, 4, 1>, gpu.known_grid_size = array<i32: 8, 8, 1>} {
      %b0 = gpu.block_id  x
      %b1 = gpu.block_id  y
      %t0 = gpu.thread_id  x
      %t1 = gpu.thread_id  y
      gpu.printf "Block (%lld, %lld, 1) - Thread (%lld, %lld, 1)\n" %b0, %b1, %t0, %t1  : index, index, index, index
      gpu.return
    }
  }
}

// Non-empty kernel should remain unchanged.
// CHECK: gpu.func @non_empty_kernel(%arg0: memref<32x32xf32>)

// -----

module attributes {gpu.container_module} {
  func.func @entry(%arg0: memref<8x8xf32>) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    gpu.launch_func  @gpu_modules::@gpu_kernel
      blocks in (%c8, %c1, %c1) threads in (%c1, %c1, %c1)
      args(%arg0 : memref<8x8xf32>)
    return
  }
  gpu.module @gpu_modules {
    gpu.func @gpu_kernel(%arg0: memref<8x8xf32>) kernel {
      gpu.return
    }

    gpu.func @gpu_func(%arg0: memref<8x8xf32>) {
      gpu.return
    }
  }
}

// CHECK: func.func @entry(%[[arg0:.+]]: memref<8x8xf32>)
// CHECK:         %[[collapse:.+]] = memref.collapse_shape %[[arg0]]{{.*}} memref<8x8xf32> into memref<64xf32>
// CHECK:         gpu.launch_func  @gpu_modules::@gpu_kernel{{.*}} args(%[[collapse]] : memref<64xf32>)
// CHECK: gpu.module @gpu_modules
// CHECK: gpu.func @gpu_kernel(%arg0: memref<64xf32>)
// CHECK: gpu.func @gpu_func(%arg0: memref<8x8xf32>)
