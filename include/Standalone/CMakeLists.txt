add_subdirectory(Dialect)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name TppCompiler)
add_public_tablegen_target(TPPCompilerPassIncGen)
add_mlir_doc(Passes TppCompilerPasses ./ -gen-pass-doc)
