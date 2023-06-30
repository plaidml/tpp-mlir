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

set(TPP_GPU "" CACHE STRING "Enables GPU runtime (default: '')")
set_property(CACHE TPP_GPU PROPERTY STRINGS "" "cuda" "vulkan")

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
include(openmp)

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
  add_subdirectory(benchmarks)

  set(BENCHMARK_DIR "${PROJECT_SOURCE_DIR}/benchmarks")
  set(CONFIG_DIR "${BENCHMARK_DIR}/config")

  # Run a small set of benchmarks with small iterations to test the benchmarks and run locally on small machines
  add_custom_target(quick-bench ${BENCHMARK_DIR}/driver.py -v --build ${PROJECT_BINARY_DIR} -n 10
                    -c ${CONFIG_DIR}/base/base.json
                    DEPENDS tpp-opt tpp-run xsmm_dnn_mlp
                    WORKING_DIRECTORY ${BENCHMARK_DIR}
                    COMMENT Run Quick Benchmarks)

  # Run performance benchmarks with small iterations to test the benchmarks and run locally on small machines
  set(MATMUL_PERF_CFGS
    ${CONFIG_DIR}/matmul/256x1024x1024.json
    ${CONFIG_DIR}/matmul/256x1024x4096.json
    ${CONFIG_DIR}/matmul/256x4096x1024.json
    ${CONFIG_DIR}/matmul/128x1024x4096.json
    ${CONFIG_DIR}/matmul/128x4096x1024.json
    ${CONFIG_DIR}/matmul/128x1024x1024.json
    ${CONFIG_DIR}/matmul/256x768x768.json
    ${CONFIG_DIR}/matmul/128x768x768.json
    ${CONFIG_DIR}/matmul/128x3072x768.json
    ${CONFIG_DIR}/matmul/128x768x3072.json
    ${CONFIG_DIR}/matmul/256x3072x768.json
    ${CONFIG_DIR}/matmul/256x768x3072.json
    ${CONFIG_DIR}/matmul/128x768x2304.json
    ${CONFIG_DIR}/matmul/1024x2560x1024.json
    ${CONFIG_DIR}/matmul/1024x1024x512.json
    ${CONFIG_DIR}/matmul/1024x352x512.json
    ${CONFIG_DIR}/matmul/1024x512x256.json
  )
  set(FC_PERF_CFGS
    ${CONFIG_DIR}/fc/256x1024x1024.json
    ${CONFIG_DIR}/fc/256x1024x4096.json
    ${CONFIG_DIR}/fc/256x4096x1024.json
    ${CONFIG_DIR}/fc/128x1024x4096.json
    ${CONFIG_DIR}/fc/128x4096x1024.json
    ${CONFIG_DIR}/fc/128x1024x1024.json
    ${CONFIG_DIR}/fc/256x768x768.json
    ${CONFIG_DIR}/fc/128x768x768.json
    ${CONFIG_DIR}/fc/128x3072x768.json
    ${CONFIG_DIR}/fc/128x768x3072.json
    ${CONFIG_DIR}/fc/256x3072x768.json
    ${CONFIG_DIR}/fc/256x768x3072.json
    ${CONFIG_DIR}/fc/128x768x2304.json
    ${CONFIG_DIR}/fc/1024x2560x1024.json
    ${CONFIG_DIR}/fc/1024x1024x512.json
    ${CONFIG_DIR}/fc/1024x352x512.json
    ${CONFIG_DIR}/fc/1024x512x256.json
  )
  string(JOIN ',' PERF_CFGS_STR ${MATMUL_PERF_CFGS} ${FC_PERF_CFGS})
  add_custom_target(quick-perf ${BENCHMARK_DIR}/driver.py -v --build ${PROJECT_BINARY_DIR} -n 10
    -c ${PERF_CFGS_STR}
                    DEPENDS tpp-opt tpp-run xsmm_dnn_mlp
                    WORKING_DIRECTORY ${BENCHMARK_DIR}
                    COMMENT Run Quick Performance Benchmarks)

  # Run baseline benchmarks with default iterations to track simple performance
  set(BENCH_CFGS
    ${CONFIG_DIR}/base/base.json
    ${CONFIG_DIR}/omp/dnn-fp32.json
    ${CONFIG_DIR}/omp/dnn-bf16.json
    ${CONFIG_DIR}/omp/mlir-fp32.json
    ${CONFIG_DIR}/omp/mlir-bf16.json
  )
  string(JOIN ',' BENCH_CFGS_STR ${BENCH_CFGS})
  add_custom_target(benchmarks ${BENCHMARK_DIR}/driver.py -v --build ${PROJECT_BINARY_DIR}
    -c ${BENCH_CFGS_STR}
                    DEPENDS tpp-opt tpp-run xsmm_dnn_mlp
                    WORKING_DIRECTORY ${BENCHMARK_DIR}
                    COMMENT Run Benchmarks)

  if (TPP_GPU)
    if (TPP_GPU MATCHES "cuda")
      set(BENCH_CUDA_CFGS
        ${CONFIG_DIR}/GPU/cuda.json
      )
      set(BENCH_CUDA_DEPS
        bench_cuda_matmul
      )
    endif()

    string(JOIN ',' BENCH_GPU_CFGS_STR ${BENCH_CUDA_CFGS})
    string(JOIN ' ' BENCH_GPU_DEPS_STR ${BENCH_CUDA_DEPS})

    # Run a small set of benchmarks with small iterations to test the benchmarks and run locally on small machines
    add_custom_target(quick-gpu-bench ${BENCHMARK_DIR}/driver.py -v --build ${PROJECT_BINARY_DIR} -n 10
    -c ${BENCH_GPU_CFGS_STR}
    DEPENDS tpp-opt tpp-run ${BENCH_GPU_DEPS_STR}
    WORKING_DIRECTORY ${BENCHMARK_DIR}
    COMMENT Run Quick GPU Benchmarks)
  endif()
endif()

# Code reformat
set(CI_DIR "${PROJECT_SOURCE_DIR}/scripts/ci")
add_custom_target(reformat-python ${CI_DIR}/lint-python.sh -i)
add_custom_target(reformat-clang  ${CI_DIR}/lint-clang.sh -i)
add_custom_target(reformat DEPENDS reformat-clang reformat-python)

# Code format checks
add_custom_target(check-python ${CI_DIR}/lint-python.sh)
add_custom_target(check-clang  ${CI_DIR}/lint-clang.sh)
add_custom_target(check-format DEPENDS check-clang check-python)

# Check-all
add_custom_target(check-all DEPENDS check check-format)

message(STATUS "TPP Project CMakeLists.txt END")
