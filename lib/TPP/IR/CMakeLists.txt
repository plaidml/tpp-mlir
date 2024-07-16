add_mlir_library(TPPIR
  MatcherUtils.cpp
  StructuredOpMatcher.cpp
  TilingUtils.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TPP

  DEPENDS
    MLIRLinalgDialect
)
