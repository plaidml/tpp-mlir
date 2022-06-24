add_subdirectory(Dialect)

set(LLVM_TARGET_DEFINITIONS TppPasses.td)
mlir_tablegen(TppPasses.h.inc -gen-pass-decls -name Tpp)
add_public_tablegen_target(MLIRTppPassIncGen)
add_mlir_doc(TppPasses TppPasses ./ -gen-pass-doc)
