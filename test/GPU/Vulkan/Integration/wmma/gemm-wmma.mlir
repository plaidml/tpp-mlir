// RUN: ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0:${ASAN_OPTIONS} \
// RUN: tpp-run %s -gpu=vulkan -gpu-wmma -print \
// RUN:  -entry-point-result=void -e entry 2>&1 | \
// RUN: FileCheck %s

module attributes {gpu.container_module} {
  func.func @entry(%arg0: memref<16x16xf16>, %arg1: memref<16x16xf16>, %arg2: memref<16x16xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    gpu.launch_func  @entry_kernel::@entry_kernel blocks in (%c1, %c1, %c1) threads in (%c32, %c1, %c1)
    args(%c1 : index, %c0 : index, %arg0 : memref<16x16xf16>, %arg1 : memref<16x16xf16>, %arg2 : memref<16x16xf16>)

    return
  }

  gpu.module @entry_kernel {
    gpu.func @entry_kernel(%arg0: index, %arg1: index, %arg2: memref<16x16xf16>, %arg3: memref<16x16xf16>, %arg4: memref<16x16xf16>) kernel
    attributes {gpu.known_block_size = array<i32: 32, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>} {
      %c0 = arith.constant 0 : index

      %C = gpu.subgroup_mma_load_matrix %arg4[%c0, %c0]
              {leadDimension = 16 : index}
              : memref<16x16xf16>
              -> !gpu.mma_matrix<16x16xf16, "COp">
      %A = gpu.subgroup_mma_load_matrix %arg2[%c0, %c0]
                {leadDimension = 16 : index}
                : memref<16x16xf16>
                -> !gpu.mma_matrix<16x16xf16, "AOp">
      %B = gpu.subgroup_mma_load_matrix %arg3[%c0, %c0]
                {leadDimension = 16 : index}
                : memref<16x16xf16>
                -> !gpu.mma_matrix<16x16xf16, "BOp">

      %R = gpu.subgroup_mma_compute %A, %B, %C
            : !gpu.mma_matrix<16x16xf16, "AOp">,
              !gpu.mma_matrix<16x16xf16, "BOp">
            -> !gpu.mma_matrix<16x16xf16, "COp">

      gpu.subgroup_mma_store_matrix %R, %arg4[%c0, %c0]
        {leadDimension = 16 : index}
        : !gpu.mma_matrix<16x16xf16, "COp">,
          memref<16x16xf16>

      gpu.return
    }
  }
}

// CHECK: ( ( 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65 ), ( 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65 ), ( 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65 ), ( 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65 ), ( 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65 ), ( 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65 ), ( 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65 ), ( 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65 ), ( 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65 ), ( 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65 ), ( 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65 ), ( 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65 ), ( 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65 ), ( 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65 ), ( 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65 ), ( 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65 ) )
