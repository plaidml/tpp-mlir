add_mlir_dialect_library(TPPVNNIDialect
    # Ops and dialect
    VNNIDialect.cpp
    VNNIOps.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Standalone

  DEPENDS
    # add_mlir_dialect macro force-prefixes with MLIR
    MLIRVNNIOpsIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
)

target_include_directories(TPPVNNIDialect
  PUBLIC
  $<BUILD_INTERFACE:${TPP_GEN_INCLUDE_DIR}>
  $<BUILD_INTERFACE:${TPP_MAIN_INCLUDE_DIR}>
)
