add_mlir_dialect_library(TritonIR
  Dialect.cpp
  Ops.cpp
  Traits.cpp
  Types.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TPP

  DEPENDS
  TritonTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
)

target_include_directories(TritonIR
  PUBLIC
  $<BUILD_INTERFACE:${TPP_GEN_INCLUDE_DIR}>
  $<BUILD_INTERFACE:${TPP_MAIN_INCLUDE_DIR}>
)
