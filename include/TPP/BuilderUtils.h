//===- Builder Utils - Helper for builder patterns ------------------------===//
// Utilities to help build MLIR
//
//===----------------------------------------------------------------------===//
#ifndef TPP_BUILDER_UTILS_H
#define TPP_BUILDER_UTILS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include "TPP/TensorInit.h"

namespace mlir {
class DenseElementsAttr;
class ModuleOp;
class MemRefType;
class TensorType;
class ShapedType;
class Type;
class TypeRange;
class OpBuilder;
class Operation;
class Value;
namespace func {
class FuncOp;
} // namespace func

// Creates a function, its entry basic block and sets the entry point
// inside that block
func::FuncOp createFunction(OpBuilder &builder, ModuleOp module,
                            llvm::StringRef name, TypeRange args,
                            TypeRange ret);

// Create a local constant dense tensor
Value createDenseTensor(OpBuilder &, TensorInitType, TensorType, int);

// Create a global dense memref
Value createDenseMemref(OpBuilder &, ModuleOp, TensorInitType, MemRefType, int);

// Return a ConstantOp of a certain type with a certain initializer
Value getConstIndex(OpBuilder &, int);
Value getConstInt(OpBuilder &, int, int);
Value getConstFloat(OpBuilder &, float, int);

} // namespace mlir

#endif // TPP_BUILDER_UTILS_H
