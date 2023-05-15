get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRToLLVMIRTranslationRegistration
        MLIROptLib
        MLIRTPP
        tpp_c_runner_utils
        )
add_llvm_executable(tpp-opt tpp-opt.cpp)

llvm_update_compile_flags(tpp-opt)
target_link_libraries(tpp-opt PRIVATE ${LIBS})
install(TARGETS tpp-opt RUNTIME DESTINATION bin)
