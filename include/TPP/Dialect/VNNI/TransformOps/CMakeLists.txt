set(LLVM_TARGET_DEFINITIONS VNNITransformOps.td)
mlir_tablegen(VNNITransformOps.h.inc -gen-op-decls)
mlir_tablegen(VNNITransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(TPPVNNITransformOpsIncGen)

add_mlir_doc(VNNITransformOps VNNIStructuredTransformOps TPP/ -gen-op-doc)
