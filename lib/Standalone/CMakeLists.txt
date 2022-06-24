add_mlir_dialect_library(MLIRStandalone
        # Ops and dialects
        Dialect/Tpp/TppDialect.cpp
        Dialect/Tpp/TppOps.cpp
        Dialect/Mathx/MathxDialect.cpp
        Dialect/Mathx/MathxOps.cpp
        Dialect/Stdx/StdxOps.cpp
        Dialect/Stdx/StdxDialect.cpp
       
        # Passes 
        LinalgMapToTpp.cpp
        TppEnforcePrecondition.cpp
        TppUtils.cpp
        TppCompiler.cpp
        CopyRemoval.cpp

        # Conversions
        TppConvertToVector.cpp
        LinalgConvertToTpp.cpp
        TppConvertToLoop.cpp  
        TppConvertToXsmm.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Standalone

        DEPENDS
        MLIRTppOpsIncGen
        MLIRMathxOpsIncGen
        MLIRTppPassIncGen

	LINK_LIBS PUBLIC
	MLIRIR
  MLIRInferTypeOpInterface
	)
