add_subdirectory(Dialect)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name TppCompiler)
add_public_tablegen_target(TPPCompilerPassIncGen)
add_mlir_doc(Passes TppCompilerPasses ./ -gen-pass-doc)

set(LLVM_TARGET_DEFINITIONS PassBundles.td)
mlir_tablegen(PassBundles.h.inc -gen-pass-decls -name TppPassBundle)
add_public_tablegen_target(TPPCompilerPassBundleIncGen)
add_mlir_doc(PassBundles TppCompilerPassBundles ./ -gen-pass-doc)
