add_subdirectory(Dialect)
add_subdirectory(IR)
add_subdirectory(GPU)

get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

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
    ConstantFoldPack.cpp
    ConvInitSimplify.cpp
    CombineTpp.cpp
    RewriteBatchMatmulToMatmul.cpp
    LinalgDeGeneralize.cpp
    DefaultPipeline.cpp
    DecomposeAggregatedOps.cpp
    ConvertForAllToParallelOp.cpp 
    LowerPacksAndUnpacks.cpp

  # Utils
    MatcherUtils.cpp
    TensorInit.cpp
    TensorInitFloat.cpp
    TensorInitInt.cpp
    BuilderUtils.cpp
    TransformUtils.cpp
    ValueUtils.cpp
    VNNIUtils.cpp

  # Conversions
    ConvertLinalgToTpp.cpp
    ConvertTppToLoops.cpp
    ConvertTppToXsmm.cpp
    ConvertXsmmToFunc.cpp
    ConvertCheckToLoops.cpp
    ConvertPerfToLoops.cpp
    ConvertPerfToFunc.cpp
    ConvertLinalgToFunc.cpp
    ConvertLinalgToXsmm.cpp
    ConvertMemRefToXsmm.cpp

  # Test Passes
    TestMatchers.cpp
    TestForToForAllRewrite.cpp

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
    TPPGPU

    MLIRIR
    MLIRInferTypeOpInterface
    ${mlir_dialect_libs}
    ${conversion_libs}
)

target_include_directories(MLIRTPP
  PUBLIC
    $<BUILD_INTERFACE:${XSMM_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${TPP_GEN_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${TPP_MAIN_INCLUDE_DIR}>
)
