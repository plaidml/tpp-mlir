add_mlir_dialect(XsmmOps xsmm)
add_mlir_doc(XsmmDialect XsmmDialect TPP/ -gen-dialect-doc)
add_mlir_doc(XsmmOps XsmmOps TPP/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS XsmmEnum.td)
mlir_tablegen(XsmmEnum.h.inc -gen-enum-decls)
mlir_tablegen(XsmmEnum.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRXsmmAttrDefIncGen)
