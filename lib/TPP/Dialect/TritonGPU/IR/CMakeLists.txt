add_mlir_dialect_library(TritonGPUIR
  Dialect.cpp
  LinearLayoutConversions.cpp
  Types.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TPP

  DEPENDS
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRGPUDialect
  TritonIR
  TritonTools
)

target_include_directories(TritonGPUIR
  PUBLIC
  $<BUILD_INTERFACE:${TPP_GEN_INCLUDE_DIR}>
  $<BUILD_INTERFACE:${TPP_MAIN_INCLUDE_DIR}>
)
