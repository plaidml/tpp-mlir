add_subdirectory(Dialect)

add_mlir_dialect(StdxOps stdx)
add_mlir_doc(StdxDialect StdxDialect Standalone/ -gen-dialect-doc)
add_mlir_doc(StdxOps StdxOps Standalone/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS TppPasses.td)
mlir_tablegen(TppPasses.h.inc -gen-pass-decls -name Tpp)
add_public_tablegen_target(MLIRTppPassIncGen)
add_mlir_doc(TppPasses TppPasses ./ -gen-pass-doc)
