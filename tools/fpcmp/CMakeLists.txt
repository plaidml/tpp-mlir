set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

add_llvm_executable(fpcmp
  fpcmp.c)

llvm_update_compile_flags(fpcmp)

install(TARGETS fpcmp RUNTIME DESTINATION bin)
