# Use LIBXSMM (make PREFIX=/path/to/libxsmm) given by LIBXSMMROOT
set(LIBXSMMROOT $ENV{LIBXSMMROOT})
# Fetch LIBXSMM (even if LIBXSMMROOT is present)
set(LIBXSMMFETCH $ENV{LIBXSMMFETCH})

if(LIBXSMMROOT AND NOT LIBXSMMFETCH)
  message(STATUS "Found LIBXSMM (${LIBXSMMROOT})")
  file(GLOB _GLOB_XSMM_SRCS LIST_DIRECTORIES false CONFIGURE_DEPENDS ${LIBXSMMROOT}/include/libxsmm/*.c)
  list(REMOVE_ITEM _GLOB_XSMM_SRCS ${LIBXSMMROOT}/include/libxsmm/libxsmm_generator_gemm_driver.c)
else()
  message(STATUS "Fetching LIBXSMM")
  include(FetchContent)

  FetchContent_Declare(
    xsmm
    URL https://github.com/libxsmm/libxsmm/archive/4ed10f7a3d74ce0947ce014f9aaf82cac918ca02.tar.gz
    URL_HASH SHA256=4b62513cfa44dd864f81ffe2ebe7e89ed10445947019e455a0e60daefbf0015b
  )

  FetchContent_GetProperties(xsmm)
  if(NOT xsmm_POPULATED)
    FetchContent_Populate(xsmm)
  endif()

  set(LIBXSMMROOT ${xsmm_SOURCE_DIR})
  file(GLOB _GLOB_XSMM_SRCS LIST_DIRECTORIES false CONFIGURE_DEPENDS ${LIBXSMMROOT}/src/*.c)
  list(REMOVE_ITEM _GLOB_XSMM_SRCS ${LIBXSMMROOT}/src/libxsmm_generator_gemm_driver.c)
endif()
set(XSMM_INCLUDE_DIRS ${LIBXSMMROOT}/include)

add_library(xsmm STATIC ${_GLOB_XSMM_SRCS})
target_include_directories(xsmm PUBLIC ${XSMM_INCLUDE_DIRS})
target_compile_definitions(xsmm PUBLIC
  LIBXSMM_DEFAULT_CONFIG
)
target_compile_definitions(xsmm PRIVATE
  __BLAS=0
)

#target_link_libraries(xsmm PUBLIC m)
