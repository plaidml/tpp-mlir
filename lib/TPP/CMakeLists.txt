add_subdirectory(Dialect)
add_subdirectory(Conversion)
add_subdirectory(IR)
add_subdirectory(GPU)
add_subdirectory(Transforms)

get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_mlir_library(MLIRTPP
  # Passes 
  DefaultPipeline.cpp
  DefaultTppPasses.cpp
  
  # Utils
  BuilderUtils.cpp
  TensorInit.cpp
  TensorInitFloat.cpp
  TensorInitInt.cpp
  TransformUtils.cpp
  ValueUtils.cpp
  VNNIUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/TPP

  DEPENDS
    TPPCompilerPassIncGen

  LINK_LIBS PUBLIC
    TPPTransforms
    TPPGPU
    MLIRIR
    ${conversion_libs}
)

target_include_directories(MLIRTPP
  PUBLIC
    $<BUILD_INTERFACE:${XSMM_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${TPP_GEN_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${TPP_MAIN_INCLUDE_DIR}>
)
