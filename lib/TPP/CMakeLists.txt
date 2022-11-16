add_subdirectory(Dialect)

add_mlir_library(MLIRTPP
  # Passes 
    ToBlockLayoutAndBack.cpp
    MapToBatchReduceGEMM.cpp
    TransformDialectInterpreter.cpp
    IteratorCollapsing.cpp
    MapConvToMatmul.cpp

  # Utils
    TransformUtils.cpp

  # Conversions
    ConvertLinalgXToLoops.cpp
    ConvertTppToLoops.cpp
    ConvertTppToXsmm.cpp
    ConvertXsmmToFunc.cpp
    ConvertCheckToFunc.cpp
    ConvertCheckToLoops.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TPP

  DEPENDS
    TPPCompilerPassIncGen
    TPPLinalgXTransformOps

    LINK_LIBS PUBLIC
    TPPLinalgXDialect
    TPPTppDialect
    TPPXsmmDialect

    MLIRIR
    MLIRInferTypeOpInterface
)

target_include_directories(MLIRTPP
  PUBLIC
    $<BUILD_INTERFACE:${TPP_GEN_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${TPP_MAIN_INCLUDE_DIR}>
)
