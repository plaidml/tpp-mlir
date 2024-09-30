add_mlir_dialect_library(TritonNvidiaGPUIR
  Dialect.cpp
  Ops.cpp
  Types.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TPP

  DEPENDS
  TritonNvidiaGPUTableGen
  TritonNvidiaGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
)

target_include_directories(TritonNvidiaGPUIR
  PUBLIC
  $<BUILD_INTERFACE:${TPP_GEN_INCLUDE_DIR}>
  $<BUILD_INTERFACE:${TPP_MAIN_INCLUDE_DIR}>
)
