add_mlir_conversion_library(MLIRMemRefToXSMM
  ConvertMemRefToXsmm.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/TPP

  DEPENDS
  TPPCompilerPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  TPPXsmmDialect
  MLIRMemRefDialect
  )
