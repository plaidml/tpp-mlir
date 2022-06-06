add_mlir_dialect_library(MLIRStandalone
        TppDialect.cpp
        TppOps.cpp
        MathxDialect.cpp
        MathxOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Standalone

        DEPENDS
        MLIRTppOpsIncGen
        MLIRMathxOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
  MLIRInferTypeOpInterface
	)
