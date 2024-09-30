set(CMAKE_COMPILE_WARNING_AS_ERROR OFF)
add_definitions(-w)

add_mlir_library(TritonTools
  LinearLayout.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TPP

  DEPENDS

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  f2reduce
)
