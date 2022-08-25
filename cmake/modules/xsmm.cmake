message(STATUS "Fetching xsmm")
include(FetchContent)

FetchContent_Declare(
  xsmm
  URL https://github.com/chelini/libxsmm/archive/8b86bb830f670acf565ed26d232d06a4f5e810ce.tar.gz
  URL_HASH SHA256=3a17d4a8d0d2f05322c02a51c64cc691ddda0bc57d27f1b7c827726f5d051e2c
)

FetchContent_GetProperties(xsmm)
if(NOT xsmm_POPULATED)
  FetchContent_Populate(xsmm)
endif()

file(GLOB _GLOB_XSMM_SRCS LIST_DIRECTORIES false CONFIGURE_DEPENDS ${xsmm_SOURCE_DIR}/src/*.c)
list(REMOVE_ITEM _GLOB_XSMM_SRCS ${xsmm_SOURCE_DIR}/src/libxsmm_generator_gemm_driver.c)

add_library(xsmm STATIC ${_GLOB_XSMM_SRCS})
target_include_directories(xsmm PUBLIC ${xsmm_SOURCE_DIR}/include)
target_compile_definitions(xsmm PUBLIC
  LIBXSMM_DEFAULT_CONFIG
)
target_compile_definitions(xsmm PRIVATE
  __BLAS=0
)

set(XSMM_INCLUDE_DIRS ${xsmm_SOURCE_DIR}/include)
