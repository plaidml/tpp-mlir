set(LIBS
      LLVMSupport
  )

set(LLVM_LINK_COMPONENTS
  Core
  Support
  native
  )

add_llvm_executable(bench_cuda_matmul
  matmul-cuda.cpp)

llvm_update_compile_flags(bench_cuda_matmul)

find_package(CUDA)

set(CUDA_LIBS
  ${CUDA_LIBRARIES}
  cublas
)

target_include_directories(bench_cuda_matmul PRIVATE ${BENCH_REF_INCLUDE_DIR})
target_link_libraries(bench_cuda_matmul PRIVATE
  ${LIBS}
  ${CUDA_LIBS}
)

install(TARGETS bench_cuda_matmul)
