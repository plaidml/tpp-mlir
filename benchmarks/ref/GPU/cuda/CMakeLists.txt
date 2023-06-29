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

target_include_directories(bench_cuda_matmul PRIVATE ${BENCH_REF_INCLUDE_DIR})
target_link_libraries(bench_cuda_matmul PRIVATE
  ${LIBS}
  ${CUDA_LIBRARIES}
)

install(TARGETS bench_cuda_matmul)
