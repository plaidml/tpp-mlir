// RUN: ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0:${ASAN_OPTIONS} \
// RUN: tpp-opt %s -gpu-to-cuda | FileCheck %s

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
