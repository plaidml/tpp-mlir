#ifndef _XEGPU_DIALECT_TD_INCLUDED_
#define _XEGPU_DIALECT_TD_INCLUDED_

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/CopyOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ShapedOpInterfaces.td"


// Provide a definition of the 'XeGPU' dialect in the ODS framework so that we
// can define our operations.
def XeGPUDialect : Dialect {
    // The namespace of our dialect
    let name = "xegpu";

    // A short one-line summary of our dialect.
    let summary = "The XeGPU dialect that models Intel GPU's ISA";

    // A longer description of our dialect.
    let description = [{
            The XeGPU dialect models Intel Xe ISA semantics but works at vector and
            TensorDesc data type. It provides 1:1 mappings to match Xe instructions like
            DPAS and 2D block load. The matrix size being processed at this level
            exactly matches the hardware instructions or the intrinsic supported by
            the lower-level GPU compiler.
        }];

    // The C++ namespace that the dialect class definition resides in.
    let cppNamespace = "::imex::xegpu";

    let dependentDialects = ["::mlir::memref::MemRefDialect"];

    // let extraClassDeclaration = [{
    //     mlir::Attribute parseAttribute(mlir::DialectAsmParser &parser, mlir::Type type) const;
    //     void printAttribute(mlir::Attribute attr, mlir::DialectAsmPrinter &printer) const;
    // }];

    // TODO: temporary disable it.
    let useDefaultTypePrinterParser = true;
    let useDefaultAttributePrinterParser = true;
    // let useDefaultAttributePrinterParser = false;
}

#endif //XEGPU_DIALECT
