add_subdirectory(Dialect)

get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_mlir_library(MLIRTPP
  # Passes 
    Bufferize.cpp
    ToBlockLayoutAndBack.cpp
    RewriteToBatchReduceGemm.cpp
    TransformDialectInterpreter.cpp
    IteratorCollapsing.cpp
    RewriteConvToMatmulImpl.cpp
    TileConsumerAndFuseProducers.cpp
    RewriteConvsToMatmulOrBrgemm.cpp
    DefaultTppPasses.cpp
    GeneralizeTensorPackAndUnPack.cpp
    RaiseToParallelLoop.cpp
    ConstantFoldPack.cpp
    ConvInitSimplify.cpp
    ConvertForAllToParallelOp.cpp

  # Utils
    TensorInit.cpp
    TransformUtils.cpp
    VNNIUtils.cpp

  # Conversions
    ConvertLinalgToTpp.cpp
    ConvertTppToLoops.cpp
    ConvertTppToXsmm.cpp
    ConvertXsmmToFunc.cpp
    ConvertCheckToLoops.cpp
    ConvertVNNIToTpp.cpp
    ConvertPerfToLoops.cpp
    ConvertPerfToFunc.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TPP

  DEPENDS
    TPPCompilerPassIncGen
    TPPLinalgXTransformOps

    LINK_LIBS PUBLIC
    TPPTppDialect
    TPPXsmmDialect
    TPPVNNIDialect
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
