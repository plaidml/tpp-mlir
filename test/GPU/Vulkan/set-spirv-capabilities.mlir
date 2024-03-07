// RUN: tpp-opt %s -set-spirv-capabilities=client-api=vulkan | \
// RUN: FileCheck %s --check-prefix=VULKAN

// RUN: tpp-opt %s -set-spirv-capabilities=client-api=opencl | \
// RUN: FileCheck %s --check-prefix=OPENCL

module attributes {gpu.container_module} {
  func.func @entry(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>, %arg2: memref<32x32xf32>) {
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    gpu.launch_func  @entry_kernel::@entry_kernel blocks in (%c32, %c32, %c1) threads in (%c1, %c1, %c1) args(%arg0 : memref<32x32xf32>, %arg1 : memref<32x32xf32>, %arg2 : memref<32x32xf32>, %c0 : index, %c32 : index, %c1 : index)
    return
  }
  gpu.module @entry_kernel {
    gpu.func @entry_kernel(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>, %arg2: memref<32x32xf32>, %arg3: index, %arg4: index, %arg5: index) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 32, 32, 1>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      scf.for %arg6 = %arg3 to %arg4 step %arg5 {
        %2 = memref.load %arg0[%0, %arg6] : memref<32x32xf32>
        %3 = memref.load %arg1[%arg6, %1] : memref<32x32xf32>
        %4 = memref.load %arg2[%0, %1] : memref<32x32xf32>
        %5 = arith.mulf %2, %3 : f32
        %6 = arith.addf %4, %5 : f32
        memref.store %6, %arg2[%0, %1] : memref<32x32xf32>
      }
      gpu.return
    }
  }
}

// VULKAN: module attributes {gpu.container_module} {
// VULKAN: gpu.module @entry_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader, Float16, StorageBuffer16BitAccess], [SPV_KHR_storage_buffer_storage_class, SPV_KHR_16bit_storage, SPV_NV_cooperative_matrix]>, api=Vulkan, #spirv.resource_limits<>>} {

// OPENCL: module attributes {gpu.container_module} {
// OPENCL: gpu.module @entry_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Bfloat16ConversionINTEL, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, StorageBuffer16BitAccess, VectorComputeINTEL, VectorAnyINTEL], [SPV_INTEL_bfloat16_conversion, SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_KHR_16bit_storage, SPV_NV_cooperative_matrix, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
