// RUN: ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0:${ASAN_OPTIONS} \
// RUN: tpp-run %s -gpu=vulkan \
// RUN:  -entry-point-result=void -e entry -print-mlir=mid 2>&1 | \
// RUN: FileCheck %s --check-prefix=SINGLE

// RUN: ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0:${ASAN_OPTIONS} \
// RUN: tpp-run %s -gpu=vulkan -n 2 \
// RUN:  -entry-point-result=void -e entry -print-mlir=mid 2>&1 | \
// RUN: FileCheck %s --check-prefix=BENCH

// XFAIL:*
// gpu.launch_func has missing 'MemoryEffectOpInterface' and breaks deallocation pass
// after LLVM fbb62d449c47bb0b49c0727c926373b41a8183c5

module attributes {
  gpu.container_module,
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, #spirv.resource_limits<>>
} {
  gpu.module @kernels {
    gpu.func @kernel_add(%arg0 : memref<8xf32>, %arg1 : memref<8xf32>, %arg2 : memref<8xf32>)
      kernel attributes { spirv.entry_point_abi = #spirv.entry_point_abi<workgroup_size = [1, 1, 1]>} {
      %0 = gpu.block_id x
      %1 = memref.load %arg0[%0] : memref<8xf32>
      %2 = memref.load %arg1[%0] : memref<8xf32>
      %3 = arith.addf %1, %2 : f32
      memref.store %3, %arg2[%0] : memref<8xf32>
      gpu.return
    }
  }

  func.func @entry(%arg0 : memref<8xf32>, %arg1 : memref<8xf32>, %arg2 : memref<8xf32>) {
    %cst1 = arith.constant 1 : index
    %cst8 = arith.constant 8 : index
    gpu.launch_func @kernels::@kernel_add
        blocks in (%cst8, %cst1, %cst1) threads in (%cst1, %cst1, %cst1)
        args(%arg0 : memref<8xf32>, %arg1 : memref<8xf32>, %arg2 : memref<8xf32>)

    return
  }
}

// SINGLE-LABEL: func.func @entry
// SINGLE: %[[ARG0:.+]] = memref.get_global @__wrapper_0
// SINGLE: %[[alloc0:.+]] = memref.alloc
// SINGLE: memref.copy %[[ARG0]], %[[alloc0]]
// SINGLE: %[[ARG1:.+]] = memref.get_global @__wrapper_1
// SINGLE: %[[alloc1:.+]] = memref.alloc
// SINGLE: memref.copy %[[ARG1]], %[[alloc1]]
// SINGLE: %[[ARG2:.+]] = memref.get_global @__wrapper_2
// SINGLE: %[[alloc2:.+]] = memref.alloc
// SINGLE: memref.copy %[[ARG2]], %[[alloc2]]
// SINGLE: call @_entry(%[[alloc0]], %[[alloc1]], %[[alloc2]])
// SINGLE-DAG: memref.dealloc %[[alloc0]]
// SINGLE-DAG: memref.dealloc %[[alloc1]]
// SINGLE-DAG: memref.dealloc %[[alloc2]]

// BENCH-LABEL: func.func @entry
// BENCH: %[[ARG0:.+]] = memref.get_global @__wrapper_0
// BENCH: %[[alloc0:.+]] = memref.alloc
// BENCH: memref.copy %[[ARG0]], %[[alloc0]]
// BENCH: %[[ARG1:.+]] = memref.get_global @__wrapper_1
// BENCH: %[[alloc1:.+]] = memref.alloc
// BENCH: memref.copy %[[ARG1]], %[[alloc1]]
// BENCH: %[[ARG2:.+]] = memref.get_global @__wrapper_2
// BENCH: %[[alloc2:.+]] = memref.alloc
// BENCH: memref.copy %[[ARG2]], %[[alloc2]]
// BENCH: perf.bench{{.*}}{
// BENCH:   call @_entry(%[[alloc0]], %[[alloc1]], %[[alloc2]])
// BENCH: }
// BENCH-DAG: memref.dealloc %[[alloc0]]
// BENCH-DAG: memref.dealloc %[[alloc1]]
// BENCH-DAG: memref.dealloc %[[alloc2]]
