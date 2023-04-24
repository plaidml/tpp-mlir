#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"

#include "TPP/BuilderUtils.h"

#include <optional>

namespace mlir {

namespace {
template <class ValueT>
arith::ConstantOp getConstant(OpBuilder &builder, Type type, ValueT value) {
  auto unkLoc = builder.getUnknownLoc();
  Attribute attr;
  if constexpr (std::numeric_limits<ValueT>::is_integer) {
    attr = builder.getIntegerAttr(type, value);
  } else if constexpr (llvm::is_one_of<ValueT, float, double>()) {
    attr = builder.getFloatAttr(type, value);
  }
  assert(attr && "Unsupported ConstantOp type");
  return builder.create<arith::ConstantOp>(unkLoc, type, attr);
}
} // anonymous namespace

func::FuncOp createFunction(OpBuilder &builder, ModuleOp module, StringRef name,
                            TypeRange args, TypeRange ret) {
  auto unkLoc = builder.getUnknownLoc();
  auto funcType = FunctionType::get(builder.getContext(), args, ret);
  auto func = func::FuncOp::create(unkLoc, name, funcType);
  func.setVisibility(SymbolTable::Visibility::Public);
  auto *entryBlock = func.addEntryBlock();
  builder.setInsertionPointToEnd(entryBlock);
  module.push_back(func);

  return func;
}

Value getConstInt(OpBuilder &builder, int value, int width) {
  switch (width) {
  case 32:
    return getConstant(builder, builder.getI32Type(), value);
  case 64:
    return getConstant(builder, builder.getI64Type(), value);
  default:
    assert(false && "Invalid constant integer size");
  }
}

Value getConstFloat(OpBuilder &builder, float value, int width) {
  switch (width) {
  case 16:
    return getConstant(builder, builder.getBF16Type(), value);
  case 32:
    return getConstant(builder, builder.getF32Type(), value);
  case 64:
    return getConstant(builder, builder.getF64Type(), value);
  default:
    assert(false && "Invalid constant float size");
  }
}

Value getConstIndex(OpBuilder &builder, int value) {
  return getConstant(builder, builder.getIndexType(), value);
}

Value createDenseTensor(OpBuilder &builder, TensorInitType initType,
                        TensorType type, int seed,
                        std::optional<ModuleOp> module) {
  auto unkLoc = builder.getUnknownLoc();

  if (module) {
    auto memrefType = MemRefType::get(type.getShape(), type.getElementType());
    auto data = createDenseMemref(builder, *module, initType, memrefType, seed);
    return builder.create<bufferization::ToTensorOp>(
        unkLoc, data, /*restrict=*/true, /*writable=*/true);
  }

  auto init = getTensorInit(initType, type.getElementType(), seed);
  auto floatInit = init->get(type);
  auto initVal = builder.create<arith::ConstantOp>(unkLoc, type, floatInit);
  auto outBuf = builder.create<tensor::EmptyOp>(unkLoc, type, ValueRange{});
  return builder
      .create<linalg::CopyOp>(unkLoc, initVal.getResult(), outBuf.getResult())
      .getResultTensors()[0];
}

Value createDenseMemref(OpBuilder &builder, ModuleOp module,
                        TensorInitType initType, MemRefType type, int seed) {
  auto unkLoc = builder.getUnknownLoc();
  StringRef globalName;
  // First create the global
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&module->getRegions().front().front());

    // Simple auto increment
    static unsigned order = 0;

    // Create global dense memrefs (Module insertion point)
    auto privAttr = builder.getStringAttr("private");

    // Auto incremental naming system
    std::string name = "__wrapper_" + std::to_string(order++);

    auto alignment = builder.getIntegerAttr(builder.getI64Type(), 128);
    auto init = getTensorInit(initType, type.getElementType(), seed);
    auto floatInit = init->get(type);

    // Create the global object in the Module's region
    auto global = builder.create<memref::GlobalOp>(
        unkLoc, StringRef(name), privAttr, type, floatInit,
        /*constant=*/false, alignment);
    globalName = global.getName();
  }
  // Get the created global value and use it
  // as an input to the kernel
  auto nameAttr = builder.getStringAttr(globalName);
  return builder.create<memref::GetGlobalOp>(unkLoc, type, nameAttr);
}
} // namespace mlir
