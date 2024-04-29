add_mlir_library(TPPRunner
  MLIRBench.cpp
  TppRunnerWrapper.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TPP

  DEPENDS
    ${mlir_dialect_libs}
    MLIRIR
    MLIRPass
    TPPPerfDialect
    TPPTransformsUtils
)
