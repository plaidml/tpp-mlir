add_mlir_dialect(VNNIOps vnni)
add_mlir_doc(VNNIDialect VNNIDialect TPP/ -gen-dialect-doc)
add_mlir_doc(VNNIOps VNNIOps TPP/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS VNNIInterfaces.td)
mlir_tablegen(VNNIInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(VNNIInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRVNNIInterfacesIncGen)
add_dependencies(mlir-headers MLIRVNNIInterfacesIncGen)
