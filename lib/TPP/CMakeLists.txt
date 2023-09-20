add_subdirectory(Dialect)
add_subdirectory(Conversion)
add_subdirectory(IR)
add_subdirectory(GPU)
add_subdirectory(Transforms)

get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_mlir_library(TPPPipeline
  DefaultPipeline.cpp
  DefaultTppPasses.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/TPP

  DEPENDS
    TPPCompilerPassIncGe
    TPPXeGPUDialect

  LINK_LIBS PUBLIC
    MLIRIR
    ${conversion_libs}
    TPPGPU
  )
