add_mlir_dialect_library(TritonCPUIR
  Dialect.cpp
  Ops.cpp
  Types.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TPP

  DEPENDS
  TritonCPUTableGen
  TritonCPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  TritonIR
)

target_include_directories(TritonCPUIR
  PUBLIC
  $<BUILD_INTERFACE:${TPP_GEN_INCLUDE_DIR}>
  $<BUILD_INTERFACE:${TPP_MAIN_INCLUDE_DIR}>
)
