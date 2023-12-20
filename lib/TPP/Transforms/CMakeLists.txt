add_subdirectory(Utils)

add_mlir_library(TPPTransforms
  Bufferize.cpp
  ConstantFoldPack.cpp
  ConvertForAllToParallelOp.cpp
  ConvInitSimplify.cpp
  DecomposeAggregatedOps.cpp
  LinalgDeGeneralize.cpp
  LowerPacksAndUnpacks.cpp
  RewriteBatchMatmulToMatmul.cpp
  RewriteConvsToMatmulOrBrgemm.cpp
  RewriteConvToMatmulImpl.cpp
  RewriteToBatchReduceGemm.cpp
  TileConsumerAndFuseProducers.cpp
  ToBlockLayoutAndBack.cpp
  TransformDialectInterpreter.cpp
  TransformUtils.cpp
  CombineXsmmPass.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TPP
  
  DEPENDS
    MLIRPerfOpsIncGen
    MLIRTppOpsIncGen
    MLIRTppAttrDefIncGen
    MLIRTppInterfaceIncGen
    MLIRXsmmOpsIncGen
    MLIRXsmmAttrDefIncGen
    TPPCompilerPassIncGen
    TPPLinalgXTransformOpsIncGen

  LINK_LIBS PUBLIC
    ${mlir_dialect_libs}
    MLIRBufferizationToMemRef
    MLIRBufferizationPipelines
    TPPTransformsUtils
  )
