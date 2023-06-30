find_package(CUDA)

# CUDA library compilation
enable_language(CUDA)
add_library(cuda_matmul SHARED
  MatmulCUDA.cu
)
set_target_properties(cuda_matmul
  PROPERTIES
  CUDA_RUNTIME_LIBRARY Shared
)

set(LLVM_LINK_COMPONENTS
  Core
  Support
  native
)

# Standard compilation
add_llvm_executable(bench_cuda_matmul
  MatmulRef.cpp
)

llvm_update_compile_flags(bench_cuda_matmul)

target_link_directories(bench_cuda_matmul PRIVATE /usr/local/cuda/lib64)
set(CUDA_LIBS
  ${CUDA_LIBRARIES}
  cublas
)

target_include_directories(bench_cuda_matmul PRIVATE ${BENCH_REF_INCLUDE_DIR})
target_link_libraries(bench_cuda_matmul PRIVATE
  LLVMSupport
  ${CUDA_LIBS}
  cuda_matmul
)

install(TARGETS bench_cuda_matmul)
