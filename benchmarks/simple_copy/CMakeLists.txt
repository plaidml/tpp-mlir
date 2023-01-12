set(LIBS
      LLVMSupport
  )

set(LLVM_LINK_COMPONENTS
  Core
  Support
  native
  )

add_llvm_executable(bench_simple_copy
  simple_copy.cpp)

llvm_update_compile_flags(bench_simple_copy)

target_include_directories(bench_simple_copy PRIVATE ../include)
target_link_libraries(bench_simple_copy PRIVATE ${LIBS})

install(TARGETS bench_simple_copy)
