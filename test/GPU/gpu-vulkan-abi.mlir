// RUN: tpp-opt %s -gpu-vulkan-abi -split-input-file | \
// RUN: FileCheck %s

module attributes {gpu.container_module} {
  func.func @gpu_launch_func_args(
      %arg0: memref<32xf32>,
      %arg1: memref<32x32xf32>,
      %arg2: memref<8x8x8xf32>,
      %arg3: memref<8x8x8x8xf32>,
      %arg4: memref<2x3x4x5xi32>,
      %arg5: index,
      %arg6: f32,
      %arg7: i32,
      %arg8: i16
    ) {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    gpu.launch_func  @gpu_modules::@gpu_kernel_args
      blocks in (%c4, %c1, %c1) threads in (%c4, %c1, %c1)
      args(
        %arg0: memref<32xf32>,
        %arg1: memref<32x32xf32>,
        %arg2: memref<8x8x8xf32>,
        %arg3: memref<8x8x8x8xf32>,
        %arg4: memref<2x3x4x5xi32>,
        %arg5: index,
        %arg6: f32,
        %arg7: i32,
        %arg8: i16
      )
    return
  }
  gpu.module @gpu_modules {
    gpu.func @gpu_kernel_args(
      %arg0: memref<32xf32>,
      %arg1: memref<32x32xf32>,
      %arg2: memref<8x8x8xf32>,
      %arg3: memref<8x8x8x8xf32>,
      %arg4: memref<2x3x4x5xi32>,
      %arg5: index,
      %arg6: f32,
      %arg7: i32,
      %arg8: i16
    )
    kernel attributes {gpu.known_block_size = array<i32: 4, 1, 1>, gpu.known_grid_size = array<i32: 4, 1, 1>} {
      %b = gpu.block_id  x
      %t = gpu.thread_id  y
      gpu.printf "Block %lld - Thread %lld\n" %b, %t : index, index
      gpu.return
    }
  }
}

// CHECK: func.func @gpu_launch_func_args
// CHECK:   memref.collapse_shape {{.*}} : memref<32x32xf32> into memref<1024xf32>
// CHECK:   memref.collapse_shape {{.*}} : memref<8x8x8xf32> into memref<512xf32>
// CHECK:   memref.collapse_shape {{.*}} : memref<8x8x8x8xf32> into memref<4096xf32>
// CHECK:   memref.collapse_shape {{.*}} : memref<2x3x4x5xi32> into memref<120xi32>
// CHECK:   memref.alloca() : memref<1xi32>
// CHECK:   memref.alloca() : memref<1xf32>
// CHECK:   memref.alloca() : memref<1xi32>
// CHECK:   memref.alloca() : memref<1xi16>
// CHECK: gpu.launch_func  @gpu_modules::@gpu_kernel_args{{.*}}args(
// CHECK-SAME: %{{[a-z0-9_]+}} : memref<32xf32>
// CHECK-SAME: %{{[a-z0-9_]+}} : memref<1024xf32>
// CHECK-SAME: %{{[a-z0-9_]+}} : memref<512xf32>
// CHECK-SAME: %{{[a-z0-9_]+}} : memref<4096xf32>
// CHECK-SAME: %{{[a-z0-9_]+}} : memref<120xi32>
// CHECK-SAME: %{{[a-z0-9_]+}} : memref<1xi32>
// CHECK-SAME: %{{[a-z0-9_]+}} : memref<1xf32>
// CHECK-SAME: %{{[a-z0-9_]+}} : memref<1xi32>
// CHECK-SAME: %{{[a-z0-9_]+}} : memref<1xi16>

// CHECK: gpu.module @gpu_modules
// CHECK: gpu.func @gpu_kernel_args(
// CHECK-SAME: %arg0: memref<32xf32>
// CHECK-SAME: %arg1: memref<1024xf32>
// CHECK-SAME: %arg2: memref<512xf32>
// CHECK-SAME: %arg3: memref<4096xf32>
// CHECK-SAME: %arg4: memref<120xi32>
// CHECK-SAME: %arg5: memref<1xi32>
// CHECK-SAME: %arg6: memref<1xf32>
// CHECK-SAME: %arg7: memref<1xi32>
// CHECK-SAME: %arg8: memref<1xi16>

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
