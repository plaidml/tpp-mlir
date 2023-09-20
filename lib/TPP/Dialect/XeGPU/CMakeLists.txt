add_mlir_dialect_library(TPPXeGPUDialect
  # Ops and dialects
    XeGPUOps.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TPP
    ${PROJECT_SOURCE_DIR}/include/TPP/Dialect/XeGPU

  DEPENDS
    # add_mlir_dialect macro force-prefixes with MLIR
    MLIRXeGPUOpsIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
)

target_include_directories(TPPXeGPUDialect
  PUBLIC
  $<BUILD_INTERFACE:${TPP_GEN_INCLUDE_DIR}>
  $<BUILD_INTERFACE:${TPP_MAIN_INCLUDE_DIR}>
)
