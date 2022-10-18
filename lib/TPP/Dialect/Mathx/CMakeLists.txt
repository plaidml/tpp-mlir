
add_mlir_dialect_library(TPPMathxDialect
  # Ops and dialects
    MathxDialect.cpp
    MathxOps.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TPP

  DEPENDS
    # add_mlir_dialect macro force-prefixes with MLIR
    MLIRMathxOpsIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
)

target_include_directories(TPPMathxDialect
  PUBLIC
    $<BUILD_INTERFACE:${TPP_GEN_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${TPP_MAIN_INCLUDE_DIR}>
)
