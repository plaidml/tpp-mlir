add_mlir_dialect_library(TPPXsmmDialect
  # Ops and dialects
    XsmmEnum.cpp
    XsmmDialect.cpp
    XsmmOps.cpp
    XsmmUtils.cpp
    XsmmVerify.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TPP

  DEPENDS
    # add_mlir_dialect macro force-prefixes with MLIR
    MLIRXsmmAttrDefIncGen
    MLIRXsmmOpsIncGen
    TPPCompilerPassIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
)

target_include_directories(TPPXsmmDialect
  PUBLIC
    $<BUILD_INTERFACE:${TPP_GEN_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${TPP_MAIN_INCLUDE_DIR}>
)
