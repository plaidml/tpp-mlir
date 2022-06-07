add_mlir_dialect_library(MLIRStandalone
        # Ops and dialects.
        TppDialect.cpp
        TppOps.cpp
        MathxDialect.cpp
        MathxOps.cpp
       
        # Passes. 
        LinalgMapToTpp.cpp

        # Heatmap.
        HeatMap.cpp
        HeatMapImpl.cpp

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
