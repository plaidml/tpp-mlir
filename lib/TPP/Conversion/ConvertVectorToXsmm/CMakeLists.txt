add_mlir_pdll_library (
        MLIRConvertVectorToXsmmPDLLPatternsIncGen
	ConvertVectorToXsmmPDL.pdll
	ConvertVectorToXsmmPDLLPatterns.h.inc
  
  EXTRA_INCLUDES
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
 )

add_mlir_conversion_library(TPPConvertVectorToXsmm
   ConvertVectorToXsmm.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/TPP
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}

  DEPENDS
  TPPCompilerPassIncGen
  MLIRConvertVectorToXsmmPDLLPatternsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  TPPXsmmDialect
  MLIRVectorDialect
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRPDLDialect
  MLIRPDLToPDLInterp
  MLIRPDLInterpDialect
  MLIRInferTypeOpInterface
  TPPIR
)

target_include_directories(
  TPPConvertVectorToXsmm
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles

 )
