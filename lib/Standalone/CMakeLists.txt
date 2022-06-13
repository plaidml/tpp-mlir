add_mlir_dialect_library(MLIRStandalone
        # Ops and dialects
        Dialect/Tpp/TppDialect.cpp
        Dialect/Tpp/TppOps.cpp
        Dialect/Mathx/MathxDialect.cpp
        Dialect/Mathx/MathxOps.cpp
        StdxOps.cpp
        StdxDialect.cpp
       
        # Passes 
        LinalgMapToTpp.cpp
        TppEnforcePrecondition.cpp
        TppUtils.cpp
        TppCompiler.cpp

        # Conversions
        TppConvertToVector.cpp
        LinalgConvertToTpp.cpp

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
