# Use LIBXSMM (make PREFIX=/path/to/libxsmm) given by LIBXSMMROOT
set(LIBXSMMROOT $ENV{LIBXSMMROOT})
# Fetch LIBXSMM (even if LIBXSMMROOT is present)
set(LIBXSMMFETCH $ENV{LIBXSMMFETCH})

if(LIBXSMMROOT AND NOT LIBXSMMFETCH)
  message(STATUS "Found LIBXSMM (${LIBXSMMROOT})")
  file(GLOB XSMM_SRCS LIST_DIRECTORIES false CONFIGURE_DEPENDS ${LIBXSMMROOT}/include/libxsmm/*.c)
  list(REMOVE_ITEM XSMM_SRCS ${LIBXSMMROOT}/include/libxsmm/libxsmm_generator_gemm_driver.c)
else()
  message(STATUS "Fetching LIBXSMM")
  include(FetchContent)

  FetchContent_Declare(
    xsmm
    URL https://github.com/libxsmm/libxsmm/archive/e53f85e5aaa02de48a3ca77b0549c0b3e4f8aee9.tar.gz
    URL_HASH SHA256=cba53c071431f5109c76cbd5789c803d4aadd0f3e671d12dd7da1203601845de
  )

  FetchContent_GetProperties(xsmm)
  if(NOT xsmm_POPULATED)
    FetchContent_Populate(xsmm)
  endif()

  set(LIBXSMMROOT ${xsmm_SOURCE_DIR})
endif()

if(NOT XSMM_SRCS)
  file(GLOB XSMM_SRCS LIST_DIRECTORIES false CONFIGURE_DEPENDS ${LIBXSMMROOT}/src/*.c)
  list(REMOVE_ITEM XSMM_SRCS ${LIBXSMMROOT}/src/libxsmm_generator_gemm_driver.c)
endif()

set(XSMM_INCLUDE_DIRS ${LIBXSMMROOT}/include)

add_library(xsmm STATIC ${XSMM_SRCS})
target_include_directories(xsmm PUBLIC ${XSMM_INCLUDE_DIRS})
target_compile_definitions(xsmm PRIVATE
  __BLAS=0
)
add_definitions(-DLIBXSMM_DEFAULT_CONFIG -U_DEBUG)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(xsmm PUBLIC Threads::Threads)
target_link_libraries(xsmm PUBLIC ${CMAKE_DL_LIBS})

include(CheckLibraryExists)
check_library_exists(m sqrt "" XSMM_LIBM)
if(XSMM_LIBM)
  target_link_libraries(xsmm PUBLIC m)
endif()
check_library_exists(rt sched_yield "" XSMM_LIBRT)
if(XSMM_LIBRT)
  target_link_libraries(xsmm PUBLIC rt)
endif()
#target_link_libraries(xsmm PUBLIC c)
