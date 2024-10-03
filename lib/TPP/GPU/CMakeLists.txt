add_mlir_library(TPPGPU
  Utils.cpp
  GpuPipeline.cpp
  GpuConversion.cpp
  GpuToCuda.cpp
  SetSPIRVCapabilities.cpp
  SetSPIRVAbiAttribute.cpp
  GpuDataTransfer.cpp
  GpuInlineConstants.cpp
  LinalgToXeGPU.cpp
  GpuVectorize.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TPP

  DEPENDS
    MLIRPerfOpsIncGen
    MLIRXsmmOpsIncGen
    TPPCompilerPassIncGen
    TPPCompilerPassBundleIncGen

  LINK_LIBS PUBLIC
    MLIRGPUDialect
    MLIRXeGPUDialect
    MLIRGPUTransforms
    MLIRGPUToSPIRV
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

if (DEFINED ${TPP_GPU})
  target_link_libraries(TPPGPU
    PUBLIC
    MLIRSPIRVDialect
    MLIRSPIRVTransforms
    MLIRGPUToSPIRV
  )
endif()
