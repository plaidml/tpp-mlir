add_mlir_library(TPPTransformsUtils
  BuilderUtils.cpp
  TensorInit.cpp
  TensorInitFloat.cpp
  TensorInitInt.cpp
  ValueUtils.cpp
  VNNIUtils.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TPP

  LINK_LIBS PUBLIC
    MLIRLinalgUtils
  )

target_include_directories(TPPTransformsUtils
  PUBLIC
    $<BUILD_INTERFACE:${XSMM_INCLUDE_DIRS}>
)
