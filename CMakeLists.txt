message(STATUS "TPP Project CMakeLists.txt BEGIN")

cmake_minimum_required(VERSION 3.13.4)

# CMP0116: Ninja generators transform `DEPFILE`s from `add_custom_command()`
# New in CMake 3.20. https://cmake.org/cmake/help/latest/policy/CMP0116.html
if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

project(tpp-dialect LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

set(TPP_GPU "" CACHE STRING "Enables GPU runtime (default: '')")
set_property(CACHE TPP_GPU PROPERTY STRINGS "" "cuda" "vulkan")

find_package(MLIR REQUIRED CONFIG)

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
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
include(openmp)

# libxsmm
include(xsmm)
message (STATUS "LIBXSMM Include dir: ${XSMM_INCLUDE_DIRS}")
include(xsmm-dnn)
message (STATUS "LIBXSMM-DNN Include dir: ${XSMM_DNN_INCLUDE_DIRS}")

# Sanitizers (include as late as possible aka append)
include(sanitizers)

# Sub directories
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(runtime)
add_subdirectory(tools)
add_subdirectory(test)

# Benchmarks
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

# GPU Benchmarks
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

  # Run a small set of benchmarks with small iterations to test the benchmarks and run locally on small machines
  add_custom_target(quick-gpu-bench ${BENCHMARK_DIR}/driver.py -v --build ${PROJECT_BINARY_DIR} -n 10
  -c ${BENCH_GPU_CFGS_STR}
  DEPENDS tpp-opt tpp-run ${BENCH_CUDA_DEPS}
  WORKING_DIRECTORY ${BENCHMARK_DIR}
  COMMENT Run Quick GPU Benchmarks)
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
