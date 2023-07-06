// RUN: ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0:${ASAN_OPTIONS} \
// RUN: tpp-opt %s -gpu-to-cuda -split-input-file | FileCheck %s

#map = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>

module attributes {gpu.container_module} {
  func.func @entry() {
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<8x8xf32>
    %alloc_0 = memref.alloc() : memref<8x8xf32>
    %alloc_1 = memref.alloc() : memref<8x8xf32>
    %cast = memref.cast %alloc : memref<8x8xf32> to memref<*xf32>
    gpu.host_register %cast : memref<*xf32>
    %cast_2 = memref.cast %alloc_0 : memref<8x8xf32> to memref<*xf32>
    gpu.host_register %cast_2 : memref<*xf32>
    %cast_3 = memref.cast %alloc_1 : memref<8x8xf32> to memref<*xf32>
    gpu.host_register %cast_3 : memref<*xf32>
    gpu.launch_func  @entry_kernel::@entry_kernel blocks in (%c8, %c8, %c1) threads in (%c1, %c1, %c1) args(%c1 : index, %c0 : index, %alloc : memref<8x8xf32>, %alloc_0 : memref<8x8xf32>, %alloc_1 : memref<8x8xf32>, %c8 : index)
    call @printMemrefF32(%cast_3) : (memref<*xf32>) -> ()
    memref.dealloc %alloc : memref<8x8xf32>
    memref.dealloc %alloc_0 : memref<8x8xf32>
    memref.dealloc %alloc_1 : memref<8x8xf32>
    return
  }

  gpu.module @entry_kernel {
    gpu.func @entry_kernel(%arg0: index, %arg1: index, %arg2: memref<8x8xf32>, %arg3: memref<8x8xf32>, %arg4: memref<8x8xf32>, %arg5: index) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = affine.apply #map(%0)[%arg0, %arg1]
      %3 = affine.apply #map(%1)[%arg0, %arg1]
      scf.for %arg6 = %arg1 to %arg5 step %arg0 {
        %4 = memref.load %arg2[%2, %arg6] : memref<8x8xf32>
        %5 = memref.load %arg3[%arg6, %3] : memref<8x8xf32>
        %6 = memref.load %arg4[%2, %3] : memref<8x8xf32>
        %7 = arith.mulf %4, %5 : f32
        %8 = arith.addf %6, %7 : f32
        memref.store %8, %arg4[%2, %3] : memref<8x8xf32>
      }
      gpu.return
    }
  }

  func.func private @printMemrefF32(memref<*xf32>)
}

// General conversion check.
// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @entry
// CHECK:         %[[C1:.*]] = memref.cast
// CHECK:         gpu.host_register %[[C1]]
// CHECK:         %[[C2:.*]] = memref.cast
// CHECK:         gpu.host_register %[[C2]]
// CHECK:         %[[C3:.*]] = memref.cast
// CHECK:         gpu.host_register %[[C3]]
// CHECK:         gpu.launch_func  @entry_kernel::@entry_kernel
// CHECK:         call @printMemrefF32
// CHECK:       }
// CHECK: gpu.module @entry_kernel attributes {gpu.binary = "
// CHECK-LABEL: llvm.func @entry_kernel
// CHECK-DAG:     nvvm.read
// CHECK-DAG:     llvm.mul
// CHECK-DAG:     llvm.add

// -----

module attributes {gpu.container_module} {
  func.func @entry(%arg0: memref<4x16x64x64xf32>, %arg1: memref<16x16x64x64xf32>, %arg2: memref<4x16x64x64xf32>) {
    %c64 = arith.constant 64 : index
    %c16 = arith.constant 16 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    gpu.launch_func  @entry_kernel::@entry_kernel blocks in (%c4, %c16, %c1) threads in (%c64, %c64, %c1) args(%arg0 : memref<4x16x64x64xf32>, %arg1 : memref<16x16x64x64xf32>, %arg2 : memref<4x16x64x64xf32>, %c0 : index, %c64 : index, %c1 : index, %c16 : index)
    return
  }
  gpu.module @entry_kernel {
    gpu.func @entry_kernel(%arg0: memref<4x16x64x64xf32>, %arg1: memref<16x16x64x64xf32>, %arg2: memref<4x16x64x64xf32>, %arg3: index, %arg4: index, %arg5: index, %arg6: index) kernel attributes {gpu.known_block_size = array<i32: 64, 64, 1>, gpu.known_grid_size = array<i32: 4, 16, 1>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = gpu.thread_id  x
      %3 = gpu.thread_id  y
      %subview = memref.subview %arg0[%0, 0, 0, 0] [1, 16, 64, 64] [1, 1, 1, 1] : memref<4x16x64x64xf32> to memref<16x64x64xf32, strided<[4096, 64, 1], offset: ?>>
      %subview_0 = memref.subview %arg1[%1, 0, 0, 0] [1, 16, 64, 64] [1, 1, 1, 1] : memref<16x16x64x64xf32> to memref<16x64x64xf32, strided<[4096, 64, 1], offset: ?>>
      %subview_1 = memref.subview %arg2[%0, %1, 0, 0] [1, 1, 64, 64] [1, 1, 1, 1] : memref<4x16x64x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
      scf.for %arg7 = %arg3 to %arg6 step %arg5 {
        scf.for %arg8 = %arg3 to %arg4 step %arg5 {
          %4 = memref.load %subview[%arg7, %2, %arg8] : memref<16x64x64xf32, strided<[4096, 64, 1], offset: ?>>
          %5 = memref.load %subview_0[%arg7, %arg8, %3] : memref<16x64x64xf32, strided<[4096, 64, 1], offset: ?>>
          %6 = memref.load %subview_1[%2, %3] : memref<64x64xf32, strided<[64, 1], offset: ?>>
          %7 = arith.mulf %4, %5 : f32
          %8 = arith.addf %6, %7 : f32
          memref.store %8, %subview_1[%2, %3] : memref<64x64xf32, strided<[64, 1], offset: ?>>
        }
      }
      gpu.return
    }
  }
}

// Verify if strided memrefs are lowered correctly.
// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @entry
// CHECK:         gpu.launch_func  @entry_kernel::@entry_kernel
// CHECK:       }
// CHECK: gpu.module @entry_kernel attributes {gpu.binary = "
// CHECK-LABEL: llvm.func @entry_kernel
// CHECK-NOT:     builtin.unrealized_conversion_cast
// CHECK-DAG:     nvvm.read
// CHECK-DAG:     llvm.mul
// CHECK-DAG:     llvm.add
