add_mlir_dialect(XeGPUOps xegpu)
add_mlir_doc(XeGPUOps XeGPUDialect Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS XeGPUOps.td)
mlir_tablegen(XeGPUOpsAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(XeGPUOpsAttrs.cpp.inc -gen-attrdef-defs)
mlir_tablegen(XeGPUOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(XeGPUOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(XeGPUOpsAttrsIncGen)
