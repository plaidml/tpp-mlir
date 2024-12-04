add_mlir_library(TPPOptions
  PipelineOptions.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/TPP

  LINK_LIBS PUBLIC
    LLVMSupport
  )
