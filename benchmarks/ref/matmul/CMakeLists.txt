set(LIBS
      LLVMSupport
  )

set(LLVM_LINK_COMPONENTS
  Core
  Support
  native
  )

add_llvm_executable(bench_matmul
  matmul.cpp)

llvm_update_compile_flags(bench_matmul)

target_include_directories(bench_matmul PRIVATE ../include)
target_link_libraries(bench_matmul PRIVATE ${LIBS})

install(TARGETS bench_matmul RUNTIME DESTINATION bin)
