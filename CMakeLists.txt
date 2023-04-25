message(STATUS "TPP Project CMakeLists.txt BEGIN")

cmake_minimum_required(VERSION 3.13.4)

# CMP0116: Ninja generators transform `DEPFILE`s from `add_custom_command()`
# New in CMake 3.20. https://cmake.org/cmake/help/latest/policy/CMP0116.html
if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

project(tpp-dialect LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

option(TPP_INSIDE_IREE "Enables being built inside IREE" OFF)

if (NOT TPP_INSIDE_IREE)
  find_package(MLIR REQUIRED CONFIG)

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
  message(STATUS "Enabling tpp tooling")
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
else()
  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir ) # --src-root
  set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include ) # --includedir
  set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  set(MLIR_LIBRARY_DIR ${LLVM_BINARY_DIR}/lib)
  include_directories(SYSTEM ${MLIR_INCLUDE_DIR})
  include_directories(SYSTEM ${MLIR_GENERATED_INCLUDE_DIR})
  include_directories(SYSTEM ${MLIR_TABLEGEN_OUTPUT_DIR})

  set(BACKEND_PACKAGE_STRING "${PACKAGE_STRING}")
  list(APPEND CMAKE_MODULE_PATH "${MLIR_MAIN_SRC_DIR}/cmake/modules")

  # message(STATUS "Enabling IREE tooling")
  # message(STATUS "Using MLIRConfig.cmake in: ${MLIR_MAIN_SRC_DIR}/cmake/modules")
  # message(STATUS "Using LLVMConfig.cmake in: ${LLVM_MAIN_SRC_DIR}/cmake")
  # list(APPEND CMAKE_MODULE_PATH "${MLIR_MAIN_SRC_DIR}/cmake/modules")
  # list(APPEND CMAKE_MODULE_PATH ${LLVM_MAIN_SRC_DIR}/cmake)
endif()
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

set(TPP_MAIN_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
set(TPP_GEN_INCLUDE_DIR "${PROJECT_BINARY_DIR}/include")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${TPP_MAIN_INCLUDE_DIR})
include_directories(${TPP_GEN_INCLUDE_DIR})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

set(LLVM_LIT_ARGS "-sv" CACHE STRING "lit default options")
include(sanitizers)

# libxsmm
include(xsmm)
message (STATUS "LIBXSMM Include dir: ${XSMM_INCLUDE_DIRS}")
include(xsmm-dnn)
message (STATUS "LIBXSMM-DNN Include dir: ${XSMM_DNN_INCLUDE_DIRS}")

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(runtime)

# IREE doesn't need any of these
if (NOT TPP_INSIDE_IREE)
  add_subdirectory(tools)
  add_subdirectory(test)

  set(BENCHMARK_DIR "${PROJECT_SOURCE_DIR}/benchmarks")
  set(CONFIG_DIR "${BENCHMARK_DIR}/config")
  # Run a small set of benchmarks with small iterations on a stable machine (non-turbo) to track simple performance
  # A separate CI job should track longer running and larger benchmarks elsewhere
  add_custom_target(benchmarks ${BENCHMARK_DIR}/driver.py -v --build ${PROJECT_BINARY_DIR}
    -c ${CONFIG_DIR}/ci/base.json,${CONFIG_DIR}/omp/mlir-fp32.json,${CONFIG_DIR}/omp/mlir-bf16.json,${CONFIG_DIR}/omp/dnn-fp32.json,${CONFIG_DIR}/omp/dnn-bf16.json
                    DEPENDS tpp-opt tpp-run xsmm_dnn_mlp
                    WORKING_DIRECTORY ${BENCHMARK_DIR}
                    COMMENT Run C++ and MLIR Benchmarks)
endif()

message(STATUS "TPP Project CMakeLists.txt END")
