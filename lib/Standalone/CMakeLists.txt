add_mlir_dialect_library(MLIRStandalone
        # Ops and dialects
        Dialect/Tpp/TppDialect.cpp
        Dialect/Tpp/TppOps.cpp
        Dialect/Tpp/TppUtils.cpp
        Dialect/Mathx/MathxDialect.cpp
        Dialect/Mathx/MathxOps.cpp
        Dialect/Stdx/StdxOps.cpp
        Dialect/Stdx/StdxDialect.cpp
        Dialect/Xsmm/XsmmOps.cpp
        Dialect/Xsmm/XsmmDialect.cpp
        Dialect/Xsmm/XsmmAttr.cpp
        Dialect/LinalgX/LinalgXOps.cpp
        Dialect/LinalgX/LinalgXDialect.cpp
       
        # Passes 
        LinalgMapToTpp.cpp
        PadSIMDDimMatmul.cpp
        TppCompiler.cpp
        VectorizeCopy.cpp
        PreBufferization.cpp
        MainClosure.cpp
        Bufferization.cpp
        TileConsumerAndFuseProducers.cpp
        DecomposeConvToMatmul.cpp
        MapMatmulToBRGemm.cpp

        # Conversions
        ConvertTppToVector.cpp  
        ConvertLinalgToTpp.cpp  
        ConvertTppToLoops.cpp    
        ConvertTppToXsmm.cpp    
        ConvertXsmmToFunc.cpp   
    
        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Standalone

        DEPENDS
        TppCompilerPassIncGen
        MLIRXsmmAttrDefIncGen

	LINK_LIBS PUBLIC
	MLIRIR
  MLIRInferTypeOpInterface
	)
