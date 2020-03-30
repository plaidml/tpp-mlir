get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRLoopOpsTransforms
        MLIRLoopAnalysis
        MLIRAffineTransformsTestPasses
        MLIRAnalysis
        MLIRDialect
        MLIREDSC
        MLIROptLib
        MLIRParser
        MLIRPass
        MLIRQuantizerFxpMathConfig
        MLIRQuantizerSupport
        MLIRQuantizerTransforms
        MLIRSPIRV
        MLIRSPIRVTestPasses
        MLIRSPIRVTransforms
        MLIRTransforms
        MLIRTransformUtils
        MLIRTestDialect
        MLIRTestIR
        MLIRTestPass
        MLIRTestTransforms
        MLIRSupport
        MLIRIR
        MLIROptLib
        LLVMSupport
        LLVMCore
        LLVMAsmParser

        MLIRStandalone
        )
add_llvm_executable(standalone-opt standalone-opt.cpp)

# Manually expand the target library, since our MLIR libraries
# aren't plugged into the LLVM dependency tracking. If we don't
# do this then we can't insert the CodeGen library after ourselves
llvm_expand_pseudo_components(TARGET_LIBS AllTargetsCodeGens)
# Prepend LLVM in front of every target: this is how the library
# are named with CMake
set(targets_to_link)
foreach (t ${TARGET_LIBS})
    list(APPEND targets_to_link "LLVM${t}")
endforeach (t)

llvm_update_compile_flags(standalone-opt)
target_link_libraries(standalone-opt PRIVATE ${LIBS} ${targets_to_link})
