add_mlir_library(TritonGPUTransforms
  Utility.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TPP

  DEPENDS


  LINK_LIBS PUBLIC
  MLIRTransforms
  MLIRTransformUtils
  TritonAnalysis
  TritonIR
  TritonGPUIR
  TritonNvidiaGPUIR
  MLIRTransformUtils
)
