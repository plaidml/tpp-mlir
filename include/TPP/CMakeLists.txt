add_subdirectory(Dialect)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name TppCompiler)
add_public_tablegen_target(TPPCompilerPassIncGen)
add_mlir_doc(Passes TppCompilerPasses ./ -gen-pass-doc)

set(LLVM_TARGET_DEFINITIONS Bundles.td)
mlir_tablegen(Bundles.h.inc -gen-pass-decls -name TppBundle)
add_public_tablegen_target(TPPCompilerBundleIncGen)
add_mlir_doc(Bundles TppCompilerBundles ./ -gen-pass-doc)
