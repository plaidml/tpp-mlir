add_mlir_dialect_library(MLIRStandalone
        StandaloneDialect.cpp
        StandaloneOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Standalone

        DEPENDS
        MLIRStandaloneOpsIncGen
        )

target_link_libraries(MLIRStandalone PUBLIC MLIRIR)

target_include_directories(MLIRStandalone PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_include_directories(MLIRStandalone PUBLIC "${PROJECT_BINARY_DIR}/include")
