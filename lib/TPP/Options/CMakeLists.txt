add_mlir_library(TPPOptions
  GpuOptions.cpp
  PipelineOptions.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/TPP

  LINK_LIBS PUBLIC
    LLVMSupport
  )
