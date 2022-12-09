add_mlir_dialect_library(TPPVNNITransformOps
    VNNITransformOps.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TPP
    ${PROJECT_BUILD_DIR}/include/TPP

  DEPENDS
    TPPVNNITransformOpsIncGen

  LINK_LIBS PUBLIC
    TPPVNNIDialect
)

target_include_directories(TPPVNNITransformOps
  PUBLIC
    $<BUILD_INTERFACE:${TPP_GEN_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${TPP_MAIN_INCLUDE_DIR}>
)
