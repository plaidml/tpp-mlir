add_mlir_library(TPPGPU
  GpuPipeline.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TPP

  DEPENDS
    MLIRLinalgDialect
    MLIRGPUOps
)
