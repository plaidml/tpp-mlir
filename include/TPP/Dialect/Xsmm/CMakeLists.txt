set(LLVM_TARGET_DEFINITIONS XsmmEnum.td)
mlir_tablegen(XsmmEnum.h.inc -gen-enum-decls)
mlir_tablegen(XsmmEnum.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRXsmmAttrDefIncGen)

