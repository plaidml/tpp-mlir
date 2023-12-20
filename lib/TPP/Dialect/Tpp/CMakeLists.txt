add_mlir_dialect_library(TPPTppDialect
  # Ops and dialects
    TppDialect.cpp
    TppOps.cpp
    TppAttr.cpp
    TppTraits.cpp
    TppInterface.cpp
    BufferizableOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TPP

  DEPENDS
    # add_mlir_dialect macro force-prefixes with MLIR
    MLIRTppOpsIncGen
    MLIRTppAttrDefIncGen
    TPPIR
    MLIRTppInterfaceIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
    TPPIR
)

target_include_directories(TPPTppDialect
  PUBLIC
  $<BUILD_INTERFACE:${TPP_GEN_INCLUDE_DIR}>
  $<BUILD_INTERFACE:${TPP_MAIN_INCLUDE_DIR}>
)
