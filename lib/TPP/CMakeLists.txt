add_subdirectory(Dialect)
add_subdirectory(IR)

get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_mlir_library(MLIRTPP
  # Passes 
    Bufferize.cpp
    ToBlockLayoutAndBack.cpp
    RewriteToBatchReduceGemm.cpp
    TransformDialectInterpreter.cpp
    RewriteConvToMatmulImpl.cpp
    TileConsumerAndFuseProducers.cpp
    RewriteConvsToMatmulOrBrgemm.cpp
    DefaultTppPasses.cpp
    GeneralizeTensorPackAndUnPack.cpp
    ConstantFoldPack.cpp
    ConvInitSimplify.cpp
    ConvertForAllToParallelOp.cpp
    CombineTpp.cpp
    HeapToStack.cpp

  # Utils
    TensorInit.cpp
    BuilderUtils.cpp
    TransformUtils.cpp
    VNNIUtils.cpp

  # Conversions
    ConvertLinalgToTpp.cpp
    ConvertTppToLoops.cpp
    ConvertTppToXsmm.cpp
    ConvertXsmmToFunc.cpp
    ConvertCheckToLoops.cpp
    ConvertPerfToLoops.cpp
    ConvertPerfToFunc.cpp

  # Test Passes
    TestMatchers.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TPP

  DEPENDS
    TPPCompilerPassIncGen
    TPPLinalgXTransformOps

    LINK_LIBS PUBLIC
    TPPIR
    TPPTppDialect
    TPPXsmmDialect
    TPPPerfDialect

    MLIRIR
    MLIRInferTypeOpInterface
    ${mlir_dialect_libs}
)

target_include_directories(MLIRTPP
  PUBLIC
    $<BUILD_INTERFACE:${TPP_GEN_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${TPP_MAIN_INCLUDE_DIR}>
)
