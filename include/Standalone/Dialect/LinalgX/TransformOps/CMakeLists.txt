set(LLVM_TARGET_DEFINITIONS LinalgXTransformOps.td)
mlir_tablegen(LinalgXTransformOps.h.inc -gen-op-decls)
mlir_tablegen(LinalgXTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(TPPLinalgXTransformOpsIncGen)

add_mlir_doc(LinalgXTransformOps LinalgXStructuredTransformOps Dialects/ -gen-op-doc)
