set(LIBS
      LLVMSupport
  )

set(LLVM_LINK_COMPONENTS
  Core
  Support
  native
  )

add_llvm_executable(bench_mlp
  mlp.cpp)

llvm_update_compile_flags(bench_mlp)

target_include_directories(bench_mlp PRIVATE ../include)
target_link_libraries(bench_mlp PRIVATE ${LIBS})

install(TARGETS bench_mlp RUNTIME DESTINATION bin)
