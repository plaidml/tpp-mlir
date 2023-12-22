add_mlir_conversion_library(TPPLinalgToXSMM
  ConvertLinalgToXsmm.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/TPP

  DEPENDS
  TPPCompilerPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  TPPXsmmDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRMemRefDialect
  MLIRFuncDialect
  TPPIR
  )
