add_mlir_dialect(XsmmOps xsmm)
add_mlir_doc(XsmmDialect XsmmDialect Standalone/ -gen-dialect-doc)
add_mlir_doc(XsmmOps XsmmOps Standalone/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS XsmmAttr.td)
mlir_tablegen(XsmmAttr.h.inc -gen-enum-decls)
mlir_tablegen(XsmmAttr.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRXsmmAttrDefIncGen)
