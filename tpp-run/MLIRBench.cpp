//===- MLIRBench.cpp - MLIR Benchmark Producer ----------------------------===//
//
// Producer for benchmark wrapper methods. Upon selecting a Kernel to run, maps
// the arguments, random initialize them and call the Kernel as many times as
// requested, taking measurements and printing the result in the end.
//
//===----------------------------------------------------------------------===//

#include "MLIRBench.h"

#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"

#include "TPP/Dialect/Perf/PerfDialect.h"
#include "TPP/Dialect/Perf/PerfOps.h"
#include "TPP/Passes.h"

using namespace mlir;

MLIRBench::MLIRBench(mlir::Operation *op, int seed, bool tppToLoops,
                     bool linalgToLoops, TensorInitType initType)
    : builder(op->getContext()), unkLoc(builder.getUnknownLoc()), seed(seed),
      tppToLoops(tppToLoops), linalgToLoops(linalgToLoops), initType(initType) {
  module = dyn_cast<ModuleOp>(op);
  assert(module && "expected a 'builtin.Module' op");
  auto *ctx = module->getContext();
  ctx->getOrLoadDialect<tensor::TensorDialect>();
  ctx->getOrLoadDialect<vector::VectorDialect>();
  ctx->getOrLoadDialect<scf::SCFDialect>();
  ctx->getOrLoadDialect<math::MathDialect>();
  ctx->getOrLoadDialect<bufferization::BufferizationDialect>();
  ctx->getOrLoadDialect<perf::PerfDialect>();
}

LogicalResult MLIRBench::findKernel(StringRef name) {
  auto &moduleOps = getModuleBlock().getOperations();
  if (!name.empty()) {
    // If the user passed the entry point, use it
    for (auto &op : moduleOps) {
      func::FuncOp func = dyn_cast_or_null<func::FuncOp>(op);
      if (func && func.getName().equals(name)) {
        kernel = func;
        break;
      }
    };
    // Needs to be in the func dialect, not LLVM
    if (!kernel)
      return module.emitError("Entry point " + name +
                              " not found as a func.func");

  } else if (moduleOps.size() == 1) {
    // Else, and there is only one function, use it
    kernel = dyn_cast_or_null<func::FuncOp>(moduleOps.front());
    if (!kernel)
      return module.emitError("Single function not in Func Dialect");

  } else {
    // If there is no entry function, and multiple functions, bail
    return module.emitError("No valid entry point, use mlir-cpu-runner");
  }

  // Ignore functions that return more than one result
  auto funcType = kernel.getFunctionType();
  if (funcType.getNumResults() > 1)
    return module.emitError("Multiple return values, use mlir-cpu-runner");

  return success();
}

LogicalResult MLIRBench::checkKernelSignature() {
  // If the function has no args or return values, just run it as is
  auto funcType = kernel.getFunctionType();
  if (funcType.getNumInputs() == 0 && funcType.getNumResults() == 0)
    return failure();

  return success();
}

LogicalResult MLIRBench::replaceSplatWithRandom() {
  if (!seed)
    return module.emitError("No seed for random init");

  // Only replace attribute if it's a dense splat
  auto replaceSplat = [&](ShapedType shape, Attribute attr) -> Attribute {
    // We only change float types
    auto elmTy = shape.getElementType();
    if (!elmTy.isBF16() && !elmTy.isF32())
      return attr;
    // We only change dense attributes that are splat
    auto value = dyn_cast<DenseElementsAttr>(attr);
    if (!value || !value.isSplat())
      return attr;
    // Only positive float data type (zero may be for ReLU, -1 for fill)
    auto elm = value.getSplatValue<FloatAttr>().getValueAsDouble();
    if (elm <= 0.0)
      return attr;
    // Generate a new random dense and return
    auto init = getTensorInit(initType, elmTy, seed);
    return init->get(shape);
  };

  // Memrefs are memref.global values
  for (auto &op : module->getRegion(0).getOps()) {
    auto global = dyn_cast<memref::GlobalOp>(op);
    if (!global)
      continue;
    auto newAttr = replaceSplat(global.getType(), global.getInitialValueAttr());
    global.setInitialValueAttr(newAttr);
  }

  // Tensors are arith.constant values
  for (auto &op : kernel->getRegion(0).getOps()) {
    auto constant = dyn_cast<arith::ConstantOp>(op);
    if (!constant)
      continue;
    auto newAttr = replaceSplat(constant.getType(), constant.getValueAttr());
    constant.setValueAttr(newAttr);
  }

  return success();
}

LogicalResult MLIRBench::renameKernel() {
  // Rename the entry point to something else and make the main the entry point
  // This is required because we can't change the original Name
  mainName = kernel.getName();
  auto newName = builder.getStringAttr("_" + mainName);
  kernel.setName(newName);

  return success();
}

LogicalResult MLIRBench::createKernelArgs() {
  // Clear current args and rebuild them from scratch
  kernelArgs.clear();

  // Create global dense memrefs (Module insertion point)
  auto &mainBody = getMainBlock();
  builder.setInsertionPointToStart(&mainBody);

  for (auto &ty : kernel.getArgumentTypes()) {
    auto arg = TypeSwitch<Type, llvm::Optional<Value>>(ty)
                   .Case<MemRefType>([&](auto memRefTy) {
                     // Create a memref global
                     auto name = createGlobal(memRefTy);
                     // Get the created global value and use it
                     // as an input to the kernel
                     auto nameAttr = builder.getStringAttr(name);
                     return builder.create<memref::GetGlobalOp>(
                         unkLoc, memRefTy, nameAttr);
                   })
                   .Case<TensorType>([&](auto tensorTy) {
                     // Create a dense const tensor and use it directly
                     // as an input to the kernel
                     return createDenseTensor(tensorTy);
                   })
                   .Default([&](auto t) { return std::nullopt; });

    if (!arg)
      return failure();

    kernelArgs.push_back(*arg);
  }

  builder.setInsertionPointToEnd(&mainBody);

  return success();
}

LogicalResult
MLIRBench::createGlobals(llvm::SmallVector<llvm::StringRef> &list) {
  auto funcType = kernel.getFunctionType();
  for (auto &ty : funcType.getInputs()) {
    auto shapedTy = cast<ShapedType>(ty);
    auto memRefTy =
        MemRefType::get(shapedTy.getShape(), shapedTy.getElementType());

    list.push_back(createGlobal(memRefTy));
  }

  return success();
}

LogicalResult MLIRBench::createMainWrapper() {
  // Add a `main` function (with no args/rets) to handle init/tear down
  auto funcType = builder.getFunctionType({}, {});
  main = func::FuncOp::create(unkLoc, mainName, funcType);
  main.setVisibility(SymbolTable::Visibility::Public);
  auto *entryBlock = main.addEntryBlock();
  builder.setInsertionPointToEnd(entryBlock);
  module.push_back(main);

  return success();
}

Operation *MLIRBench::callKernel() {
  // Call the kernel
  auto call = builder.create<func::CallOp>(unkLoc, kernel, kernelArgs);

  // Cleanup kernel result if the returned value is a buffer
  auto funcType = kernel.getFunctionType();
  if (funcType.getNumResults() != 0) {
    auto result = call->getOpResult(0);

    if (dyn_cast_or_null<MemRefType>(result.getType()))
      builder.create<memref::DeallocOp>(unkLoc, result);
  }

  return call;
}

Value MLIRBench::getKernelResult(Operation *kernelCall) {
  // Set the result to either the return value or the last argument, if the
  // kernel return is void.
  auto funcType = kernel.getFunctionType();

  return funcType.getNumResults() == 0 ? kernelArgs.back()
                                       : kernelCall->getOpResult(0);
}

Value MLIRBench::createTimerLoop(unsigned n) {
  // Allocates buffer for results
  auto count = getConstant(builder.getI64Type(), n);
  auto memrefType = MemRefType::get({n}, builder.getF64Type());
  auto acc = builder.create<memref::AllocOp>(unkLoc, memrefType);

  // Create perf benchmarking region, set insertion to inside the body
  auto loop = builder.create<perf::BenchOp>(unkLoc, count, acc);
  builder.setInsertionPointToStart(loop.getBody());

  // Call the kernel, ignore output
  callKernel();

  // Revert insertion point and return the accumulation ID
  builder.setInsertionPointAfter(loop);
  return acc;
}

Value MLIRBench::getTimerStats(Value acc) {
  auto callMean =
      builder.create<perf::MeanOp>(unkLoc, builder.getF64Type(), acc);
  auto mean = callMean.getMean();
  auto callDev =
      builder.create<perf::StdevOp>(unkLoc, builder.getF64Type(), acc, mean);
  auto dev = callDev.getStdev();

  // Create a vector<2xf64> so we can print
  auto zeroF = getConstant(builder.getF64Type(), 0.0);
  auto vectorType = VectorType::get({2}, builder.getF64Type());
  auto stats = builder.create<vector::SplatOp>(unkLoc, vectorType, zeroF);

  // Insert mean, dev (as a chain) into vector, return end of chain
  auto zeroIAttr = builder.getIntegerAttr(builder.getI64Type(), 0);
  auto zeroI = builder.create<arith::ConstantOp>(unkLoc, builder.getI64Type(),
                                                 zeroIAttr);
  auto insMean =
      builder.create<vector::InsertElementOp>(unkLoc, mean, stats, zeroI);
  auto oneIAttr = builder.getIntegerAttr(builder.getI64Type(), 1);
  auto oneI =
      builder.create<arith::ConstantOp>(unkLoc, builder.getI64Type(), oneIAttr);
  auto insDev =
      builder.create<vector::InsertElementOp>(unkLoc, dev, insMean, oneI);

  // Clean up results buffer
  builder.create<memref::DeallocOp>(unkLoc, acc);

  return insDev;
}

void MLIRBench::printVector(Value vector) {
  auto op = vector;
  auto vectorValue = vector.getType().dyn_cast<VectorType>();
  if (vectorValue.getElementType().isBF16()) {
    VectorType vecType =
        VectorType::get(vectorValue.getShape(), builder.getF32Type());
    op = builder.create<arith::ExtFOp>(unkLoc, vecType, vector, std::nullopt);
  }
  builder.create<vector::PrintOp>(unkLoc, op);
}

LogicalResult MLIRBench::printShapedType(mlir::Value val) {
  OpBuilder::InsertionGuard guard(builder);

  auto outputType = cast<ShapedType>(val.getType());
  assert(outputType && "expected a shaped type");

  // Read into a vector and print output
  // We don't want to alloc the whole tensor as a vector,
  // so we pick the inner dimension and iterate through the outer ones.
  VectorType vecType;
  auto lastDim = outputType.getRank() - 1;
  ArrayRef<int64_t> outerDims(1);
  if (outputType.getRank() > 1) {
    ArrayRef<int64_t> innerDims(&outputType.getShape()[lastDim], 1);
    vecType = VectorType::get(innerDims, outputType.getElementType());
    outerDims =
        ArrayRef<int64_t>(&outputType.getShape()[0], outputType.getRank() - 1);
  } else {
    vecType =
        VectorType::get(outputType.getShape(), outputType.getElementType());
  }
  assert(outerDims.size() == 1 && "Only supports 2D tensors for now");

  // Vector undefined value
  APFloat vectorFloatValue = APFloat(-1.0F);
  Value minusOne;
  if (outputType.getElementType().isBF16()) {
    bool ignored;
    vectorFloatValue.convert(APFloat::BFloat(), APFloat::rmNearestTiesToEven,
                             &ignored);

    minusOne = builder.create<arith::ConstantFloatOp>(
        unkLoc, vectorFloatValue, FloatType::getBF16(builder.getContext()));
  } else {
    minusOne = builder.create<arith::ConstantFloatOp>(unkLoc, vectorFloatValue,
                                                      builder.getF32Type());
  }

  // Loop through the shaped type, transfer each dim to vector
  auto count = getConstant(builder.getIndexType(), outerDims[0]);
  auto zero = getConstant(builder.getIndexType(), 0);
  auto one = getConstant(builder.getIndexType(), 1);
  auto loop = builder.create<scf::ForOp>(unkLoc, zero, count, one);
  builder.setInsertionPointToStart(loop.getBody());

  // Loop body
  auto beginIdx = loop.getInductionVar();
  auto vector = builder.create<vector::TransferReadOp>(
      unkLoc, vecType, val, ValueRange{beginIdx, zero}, minusOne);
  printVector(vector);

  // Finally lower to LLVM Dialect
  return success();
}

LogicalResult MLIRBench::printResult(Operation *kernelCall) {
  OpBuilder::InsertionGuard guard(builder);

  // Build print logic directly after the kernel call.
  builder.setInsertionPointAfter(kernelCall);

  return printShapedType(getKernelResult(kernelCall));
}

LogicalResult MLIRBench::finalize(bool dumpMLIR) {
  // If we created a main at all...
  // return void and add func to Module
  if (main) {
    builder.create<func::ReturnOp>(unkLoc);
  }

  if (dumpMLIR)
    module->print(llvm::outs());

  // A set of default passes that lower any input IR to LLVM
  PassManager passManager(module->getContext());
  applyPassManagerCLOptions(passManager);

  // Apply the default preprocessing pass
  passManager.addPass(tpp::createDefaultTppPass(tppToLoops, linalgToLoops));

  // Bufferization, if needed
  passManager.addNestedPass<func::FuncOp>(createTensorBufferizePass());
  passManager.addNestedPass<func::FuncOp>(vector::createVectorBufferizePass());
  passManager.addNestedPass<func::FuncOp>(createLinalgBufferizePass());

  // Partial Lowering
  passManager.addPass(memref::createExpandStridedMetadataPass());
  passManager.addPass(createLowerAffinePass());
  passManager.addPass(tpp::createConvertPerfToLoopsPass());
  passManager.addPass(tpp::createConvertPerfToFuncPass());
  passManager.addPass(createConvertTensorToLinalgPass());
  passManager.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  passManager.addPass(arith::createArithExpandOpsPass());
  passManager.addPass(createConvertVectorToSCFPass());
  passManager.addPass(createConvertSCFToCFPass());

  // Lower to LLVM
  passManager.addPass(createConvertVectorToLLVMPass());
  passManager.addPass(createConvertFuncToLLVMPass());
  passManager.addPass(createFinalizeMemRefToLLVMConversionPass());
  passManager.addPass(createConvertMathToLLVMPass());
  passManager.addNestedPass<func::FuncOp>(createArithToLLVMConversionPass());
  passManager.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  passManager.addPass(createReconcileUnrealizedCastsPass());

  auto result = passManager.run(module);
  if (failed(result)) {
    llvm::errs() << "ERROR: Failed to lower Module to LLVM dialect\n";
    module->print(llvm::errs());
  }

  return result;
}

//----------------------- Helpers & private methods

template <class ValueT>
arith::ConstantOp MLIRBench::getConstant(Type type, ValueT value) {
  Attribute attr;
  if constexpr (std::numeric_limits<ValueT>::is_integer) {
    attr = builder.getIntegerAttr(type, value);
  } else if constexpr (llvm::is_one_of<ValueT, float, double>()) {
    attr = builder.getFloatAttr(type, value);
  }
  assert(attr && "Unsupported ConstantOp type");
  return builder.create<arith::ConstantOp>(unkLoc, type, attr);
}

DenseElementsAttr MLIRBench::getDenseAttribute(ShapedType shape) {
  auto init = getTensorInit(initType, shape.getElementType(), seed);
  return init->get(shape);
}

llvm::StringRef MLIRBench::createGlobal(MemRefType type) {
  // Create global dense memrefs (Module insertion point)
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&getModuleBlock());

  // Simple auto increment
  static unsigned order = 0;

  // Create global dense memrefs (Module insertion point)
  auto privAttr = builder.getStringAttr("private");

  // Auto incremental naming system
  std::string name = "__wrapper_" + std::to_string(order++);

  auto alignment = builder.getIntegerAttr(builder.getI64Type(), 128);
  auto floatInit = getDenseAttribute(type);

  // Create the global object in the Module's region
  auto global = builder.create<memref::GlobalOp>(unkLoc, StringRef(name),
                                                 privAttr, type, floatInit,
                                                 /*constant=*/false, alignment);

  return global.getName();
}

MemRefType MLIRBench::getGlobalType(llvm::StringRef name) {
  auto op = module.lookupSymbol<memref::GlobalOp>(name);
  assert(op && "memref::Global not found");
  auto memRefTy = dyn_cast_or_null<MemRefType>(op.getType());
  assert(memRefTy && "memref::Global type not a memref?");
  return memRefTy;
}

Value MLIRBench::createDenseTensor(TensorType type) {
  auto floatInit = getDenseAttribute(type);
  return builder.create<arith::ConstantOp>(unkLoc, type, floatInit);
}

Block &MLIRBench::getModuleBlock() {
  return module->getRegions().front().front();
}

Block &MLIRBench::getMainBlock() { return main.getBody().front(); }

LogicalResult MLIRBench::emitError(llvm::Twine desc) {
  return module.emitError(desc);
}
