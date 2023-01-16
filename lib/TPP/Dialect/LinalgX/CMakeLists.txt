add_mlir_dialect_library(TPPLinalgXDialect
  # Ops and dialects
    BufferizableOpInterfaceImpl.cpp
    LinalgXDialect.cpp
    LinalgXOps.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TPP

  DEPENDS
    # add_mlir_dialect macro force-prefixes with MLIR
    MLIRLinalgXOpsIncGen
    TPPLinalgXTransformOpsIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
)

target_include_directories(TPPLinalgXDialect
  PUBLIC
    $<BUILD_INTERFACE:${TPP_GEN_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${TPP_MAIN_INCLUDE_DIR}>
)
