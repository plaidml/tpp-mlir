// RUN: ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0:${ASAN_OPTIONS} \
// RUN: tpp-run %s -gpu=vulkan \
// RUN:  -entry-point-result=void -e entry 2>&1 | \
// RUN: FileCheck %s

// XFAIL:*
// gpu.launch_func has missing 'MemoryEffectOpInterface' and breaks deallocation pass
// after LLVM fbb62d449c47bb0b49c0727c926373b41a8183c5

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
} {
  func.func @entry(%arg0: memref<8x8xf32>, %arg1: memref<8x8xf32>, %arg2: memref<8x8xf32>) {
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    gpu.launch_func  @entry_kernel::@entry_kernel blocks in (%c8, %c8, %c1) threads in (%c1, %c1, %c1) args(%arg0 : memref<8x8xf32>, %arg1 : memref<8x8xf32>, %arg2 : memref<8x8xf32>, %c0 : index, %c8 : index, %c1 : index)
    %cast = memref.cast %arg2 : memref<8x8xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    return
  }

  gpu.module @entry_kernel {
    gpu.func @entry_kernel(%arg0: memref<8x8xf32>, %arg1: memref<8x8xf32>, %arg2: memref<8x8xf32>, %arg3: index, %arg4: index, %arg5: index)
    kernel attributes { spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [1, 1, 1]> } {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      scf.for %arg6 = %arg3 to %arg4 step %arg5 {
        %2 = memref.load %arg0[%0, %arg6] : memref<8x8xf32>
        %3 = memref.load %arg1[%arg6, %1] : memref<8x8xf32>
        %4 = memref.load %arg2[%0, %1] : memref<8x8xf32>
        %5 = arith.mulf %2, %3 : f32
        %6 = arith.addf %4, %5 : f32
        memref.store %6, %arg2[%0, %1] : memref<8x8xf32>
      }
      gpu.return
    }
  }

  func.func private @printMemrefF32(memref<*xf32>)
}

// CHECK-COUNT-8: [9, 9, 9, 9, 9, 9, 9, 9]
