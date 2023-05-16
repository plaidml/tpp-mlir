add_mlir_library(TPPGPU
  GpuPipeline.cpp
  GpuToCuda.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TPP

  LINK_LIBS PUBLIC
    MLIRGPUOps
    MLIRGPUTransforms
    MLIRSCFToGPU
    MLIRGPUToNVVMTransforms
)

if (TPP_GPU)
  target_compile_definitions(obj.TPPGPU
    PRIVATE
    TPP_GPU_ENABLE=1
  )
endif()
