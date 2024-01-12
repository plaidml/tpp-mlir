add_mlir_library(TPPGPU
  Utils.cpp
  GpuPipeline.cpp
  GpuConversion.cpp
  GpuToCuda.cpp
  GpuToVulkan.cpp
  GPUToSPIRVPass.cpp
  SetSPIRVCapabilities.cpp
  SetSPIRVAbiAttribute.cpp
  GpuVulkanAbi.cpp
  LinalgToGpu.cpp
  GpuDataTransfer.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TPP

  DEPENDS
    MLIRPerfOpsIncGen
    MLIRXsmmOpsIncGen
    TPPCompilerPassIncGen
    TPPLinalgXTransformOpsIncGen

  LINK_LIBS PUBLIC
    MLIRGPUDialect
    MLIRGPUTransforms
    MLIRSCFToGPU
    MLIRSCFToSPIRV
    MLIRMathToSPIRV
    MLIRControlFlowToSPIRV
    MLIRMemRefTransforms
    TPPIR
)

if (TPP_GPU MATCHES "cuda")
  target_compile_definitions(obj.TPPGPU
    PRIVATE
    TPP_CUDA_ENABLE=1
  )

  target_link_libraries(TPPGPU
    PUBLIC
    MLIRGPUToNVVMTransforms
  )
endif()

if (TPP_GPU MATCHES "vulkan")
  target_compile_definitions(obj.TPPGPU
    PRIVATE
    TPP_VULKAN_ENABLE=1
  )

  target_link_libraries(TPPGPU
    PUBLIC
    MLIRSPIRVDialect
    MLIRSPIRVTransforms
    MLIRGPUToVulkanTransforms
    MLIRGPUToSPIRV
  )
endif()
