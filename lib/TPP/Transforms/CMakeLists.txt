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
  TransformUtils.cpp
  CombineXsmmPass.cpp
  IntelAMXTileConfig.cpp
  IntelAMXTileConfigHoisting.cpp
  LinalgConvertCompareSelectToMaximumfPass.cpp
  ConvertAddInplacePass.cpp
  LoopInsertion.cpp
  LoopExpansion.cpp
  LoopShuffle.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TPP

  DEPENDS
    MLIRPerfOpsIncGen
    MLIRXsmmOpsIncGen
    MLIRXsmmAttrDefIncGen
    TPPCompilerPassIncGen

  LINK_LIBS PUBLIC
    ${mlir_dialect_libs}
    MLIRIR
    MLIRSupport
    MLIRBufferizationToMemRef
    MLIRBufferizationPipelines
    TPPTransformsUtils
    TPPIR
    TPPXsmmDialect
    TPPCheckToLoops
    TPPLinalgToFunc
    TPPPerfToFunc
    TPPPerfToLoop
    TPPXsmmToFunc
  )
