add_mlir_conversion_library(MLIRPerfToFunc
  ConvertPerfToFunc.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/TPP

  DEPENDS
  TPPCompilerPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  TPPPerfDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRMathDialect
  )
