set(LLVM_TARGET_DEFINITIONS LinalgXTransformOps.td)
mlir_tablegen(LinalgXTransformOps.h.inc -gen-op-decls)
mlir_tablegen(LinalgXTransformOps.cpp.inc -gen-op-defs)
mlir_tablegen(LinalgXTransformOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(LinalgXTransformOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRLinalgXTransformOpsIncGen)

add_mlir_doc(LinalgXTransformOps LinalgXStructuredTransformOps Dialects/ -gen-op-doc)
