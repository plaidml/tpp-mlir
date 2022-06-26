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
       
        # Passes 
        LinalgMapToTpp.cpp
        TppEnforcePrecondition.cpp
        TppCompiler.cpp
        CopyRemoval.cpp
        VectorizeCopy.cpp

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

	LINK_LIBS PUBLIC
	MLIRIR
  MLIRInferTypeOpInterface
	)
