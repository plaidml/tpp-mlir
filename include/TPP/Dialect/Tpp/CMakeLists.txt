add_mlir_dialect(TppOps tpp)
add_mlir_doc(TppDialect TppDialect TPP/ -gen-dialect-doc)
add_mlir_doc(TppOps TppOps TPP/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS TppAttr.td)
mlir_tablegen(TppAttr.h.inc -gen-enum-decls)
mlir_tablegen(TppAttr.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRTppAttrDefIncGen)

set(LLVM_TARGET_DEFINITIONS TppInterface.td)
mlir_tablegen(TppInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(TppInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRTppInterfaceIncGen)
add_dependencies(mlir-headers MLIRTppInterfaceIncGen)
