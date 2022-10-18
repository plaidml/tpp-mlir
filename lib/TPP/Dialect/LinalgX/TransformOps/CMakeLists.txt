add_mlir_dialect_library(TPPLinalgXTransformOps
    LinalgXTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TPP

  DEPENDS
    TPPLinalgXTransformOpsIncGen

  LINK_LIBS PUBLIC
    TPPLinalgXDialect
)

target_include_directories(TPPLinalgXTransformOps
  PUBLIC
    $<BUILD_INTERFACE:${TPP_GEN_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${TPP_MAIN_INCLUDE_DIR}>
)
