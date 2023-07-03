add_mlir_library(TPPGPU
  Utils.cpp
  GpuPipeline.cpp
  GpuConversion.cpp
  GpuToCuda.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TPP

  LINK_LIBS PUBLIC
    MLIRGPUDialect
    MLIRGPUTransforms
    MLIRSCFToGPU
    MLIRGPUToNVVMTransforms
)

if (TPP_GPU MATCHES "cuda")
  target_compile_definitions(obj.TPPGPU
    PRIVATE
    TPP_CUDA_ENABLE=1
  )
endif()
