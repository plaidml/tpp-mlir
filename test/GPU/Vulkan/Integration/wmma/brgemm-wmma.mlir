// RUN: ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0:${ASAN_OPTIONS} \
// RUN: tpp-run %s -gpu=vulkan -gpu-wmma \
// RUN:  -entry-point-result=void -e entry 2>&1 | \
// RUN: FileCheck %s

#map = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>

module attributes {gpu.container_module} {
  func.func @entry(%arg0: memref<2x4x16x16xf16>, %arg1: memref<4x4x16x16xf16>, %arg2: memref<2x4x16x16xf16>) {
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    gpu.launch_func  @entry_kernel::@entry_kernel blocks in (%c2, %c4, %c1) threads in (%c32, %c1, %c1)
    args(%c1 : index, %c0 : index, %arg0 : memref<2x4x16x16xf16>, %arg1 : memref<4x4x16x16xf16>, %arg2 : memref<2x4x16x16xf16>)

    %out = memref.alloc() : memref<2x4x16x16xf16>
    %tOut = gpu.memcpy async %out, %arg2 : memref<2x4x16x16xf16>, memref<2x4x16x16xf16>
    gpu.wait [%tOut]
    %vcst = arith.constant -1.000000e+00 : f16
    %v0 = vector.transfer_read %out[%c1, %c2, %c0, %c0], %vcst : memref<2x4x16x16xf16>, vector<16x16xf16>
    vector.print %v0 : vector<16x16xf16>
    memref.dealloc %out : memref<2x4x16x16xf16>

    return
  }

  gpu.module @entry_kernel {
    gpu.func @entry_kernel(%arg0: index, %arg1: index, %arg2: memref<2x4x16x16xf16>, %arg3: memref<4x4x16x16xf16>, %arg4: memref<2x4x16x16xf16>) kernel
    attributes {gpu.known_block_size = array<i32: 32, 1, 1>, gpu.known_grid_size = array<i32: 2, 4, 1>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = affine.apply #map(%0)[%arg0, %arg1]
      %3 = affine.apply #map(%1)[%arg0, %arg1]
      %subview = memref.subview %arg2[%2, 0, 0, 0] [1, 4, 16, 16] [1, 1, 1, 1] : memref<2x4x16x16xf16> to memref<4x16x16xf16, strided<[256, 16, 1], offset: ?>>
      %subview_0 = memref.subview %arg3[%3, 0, 0, 0] [1, 4, 16, 16] [1, 1, 1, 1] : memref<4x4x16x16xf16> to memref<4x16x16xf16, strided<[256, 16, 1], offset: ?>>
      %subview_1 = memref.subview %arg4[%2, %3, 0, 0] [1, 1, 16, 16] [1, 1, 1, 1] : memref<2x4x16x16xf16> to memref<16x16xf16, strided<[16, 1], offset: ?>>

      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index

      %init = gpu.subgroup_mma_load_matrix %subview_1[%c0, %c0]
              {leadDimension = 16 : index}
              : memref<16x16xf16, strided<[16, 1], offset: ?>>
              -> !gpu.mma_matrix<16x16xf16, "COp">

      %sum = scf.for %arg6 = %c0 to %c4 step %c1 iter_args(%acc = %init) -> !gpu.mma_matrix<16x16xf16, "COp"> {
        %tile_A = gpu.subgroup_mma_load_matrix %subview[%arg6, %c0, %c0]
                  {leadDimension = 16 : index}
                  : memref<4x16x16xf16, strided<[256, 16, 1], offset: ?>>
                  -> !gpu.mma_matrix<16x16xf16, "AOp">
        %tile_B = gpu.subgroup_mma_load_matrix %subview_0[%arg6, %c0, %c0]
                  {leadDimension = 16 : index}
                  : memref<4x16x16xf16, strided<[256, 16, 1], offset: ?>>
                  -> !gpu.mma_matrix<16x16xf16, "BOp">
        %R = gpu.subgroup_mma_compute %tile_A, %tile_B, %acc
              : !gpu.mma_matrix<16x16xf16, "AOp">,
                !gpu.mma_matrix<16x16xf16, "BOp">
              -> !gpu.mma_matrix<16x16xf16, "COp">
        scf.yield %R : !gpu.mma_matrix<16x16xf16, "COp">
      }

      gpu.subgroup_mma_store_matrix %sum, %subview_1[%c0, %c0]
        {leadDimension = 16 : index}
        : !gpu.mma_matrix<16x16xf16, "COp">,
          memref<16x16xf16, strided<[16, 1], offset: ?>>

      gpu.return
    }
  }
}

// CHECK: ( ( 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65 ), ( 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65 ), ( 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65 ), ( 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65 ), ( 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65 ), ( 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65 ), ( 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65 ), ( 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65 ), ( 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65 ), ( 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65 ), ( 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65 ), ( 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65 ), ( 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65 ), ( 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65 ), ( 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65 ), ( 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65 ) )
