//===- MLIRBench.cpp - MLIR Benchmark Producer ----------------------------===//
//
// Producer for benchmark wrapper methods. Upon selecting a Kernel to run, maps
// the arguments, random initialize them and call the Kernel as many times as
// requested, taking measurements and printing the result in the end.
//
//===----------------------------------------------------------------------===//

#include "MLIRBench.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
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

using namespace mlir;

MLIRBench::MLIRBench(mlir::Operation *op)
    : builder(op->getContext()), unkLoc(builder.getUnknownLoc()) {
  module = dyn_cast<ModuleOp>(op);
  assert(module && "expected a 'builtin.Module' op");
  auto *ctx = module->getContext();
  ctx->getOrLoadDialect<tensor::TensorDialect>();
  ctx->getOrLoadDialect<vector::VectorDialect>();
  ctx->getOrLoadDialect<scf::SCFDialect>();

  // TODO: Remove this once we use the perf dialect
  declareGlobalFunctions();
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
    return module.emitError("Entry point already created, use mlir-cpu-runner");

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

LogicalResult
MLIRBench::createGlobals(llvm::SmallVector<llvm::StringRef> &list) {
  // Create global dense memrefs (Module insertion point)
  builder.setInsertionPointToStart(&getModuleBlock());
  auto funcType = kernel.getFunctionType();
  for (auto &ty : funcType.getInputs()) {
    auto memRefTy = dyn_cast_or_null<MemRefType>(ty);
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

Value MLIRBench::callKernel(llvm::SmallVector<llvm::StringRef> &list) {
  // Get those globals as arguments (function insertion point)
  // Cache locally, so we can avoid doing it again inside the loop
  if (kernelArgs.empty()) {
    for (auto &name : list) {
      // GetGlobal op properties
      auto nameAttr = builder.getStringAttr(name);
      auto type = getGlobalType(name);
      auto getGlobal =
          builder.create<memref::GetGlobalOp>(unkLoc, type, nameAttr);

      // Add argument to list
      kernelArgs.push_back(getGlobal);
    }
  }

  // Call the Kernel, making sure to set the result to either the return value
  // or the last argument, if the return is void.
  Value result;
  auto funcType = main.getFunctionType();
  if (funcType.getNumResults() == 0) {
    builder.create<func::CallOp>(unkLoc, kernel, kernelArgs);
    result = kernelArgs.back();
  } else {
    auto call = builder.create<func::CallOp>(unkLoc, kernel, kernelArgs);
    result = call->getOpResult(0);
  }

  return result;
}

Value MLIRBench::createTimerLoop(llvm::SmallVector<llvm::StringRef> &list,
                                 unsigned n) {
  // Allocates the vector for results
  auto callAlloc =
      builder.create<func::CallOp>(unkLoc, timer.alloc, ValueRange());
  auto acc = callAlloc.getResult(0);

  // Create a SCF loop, set insertion to inside loop
  auto indexType = builder.getIndexType();
  auto countAttr = builder.getIntegerAttr(indexType, n);
  auto count = builder.create<arith::ConstantOp>(unkLoc, indexType, countAttr);
  auto zeroAttr = builder.getIntegerAttr(indexType, 0);
  auto zero = builder.create<arith::ConstantOp>(unkLoc, indexType, zeroAttr);
  auto oneAttr = builder.getIntegerAttr(indexType, 1);
  auto one = builder.create<arith::ConstantOp>(unkLoc, indexType, oneAttr);
  auto loop = builder.create<scf::ForOp>(unkLoc, zero, count, one);
  builder.setInsertionPointToStart(loop.getBody());

  // Call timer_now()
  builder.create<func::CallOp>(unkLoc, timer.start, ValueRange(acc));

  // Call the kernel, ignore output
  callKernel(list);

  // Call timer_now() again
  builder.create<func::CallOp>(unkLoc, timer.stop, ValueRange(acc));

  // Revert insertion point and return the accumulation ID
  builder.setInsertionPointAfter(loop);
  return acc;
}

Value MLIRBench::getTimerStats(Value acc) {
  // Get stats (this is done once, but we can get values in separate)
  auto callMean =
      builder.create<func::CallOp>(unkLoc, timer.average, ValueRange{acc});
  auto mean = callMean.getResult(0);
  auto callDev =
      builder.create<func::CallOp>(unkLoc, timer.deviation, ValueRange{acc});
  auto dev = callDev.getResult(0);

  // Create a vector<2xf64> so we can print
  auto zeroFAttr = builder.getFloatAttr(builder.getF64Type(), 0.0);
  auto zeroF = builder.create<arith::ConstantOp>(unkLoc, builder.getF64Type(),
                                                 zeroFAttr);
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

  return insDev;
}

void MLIRBench::printVector(Value vector) {
  auto op = vector;
  if (vector.getType().dyn_cast<VectorType>().getElementType().isBF16()) {
    SmallVector<int64_t> dims;
    for (int i = 0; i < vector.getType().dyn_cast<VectorType>().getRank();
         i++) {
      dims.push_back(vector.getType().dyn_cast<VectorType>().getShape()[i]);
    }
    VectorType vecType = VectorType::get(dims, builder.getF32Type());
    op = builder.create<arith::ExtFOp>(unkLoc, vecType, vector, std::nullopt);
    op.dump();
  }
  builder.create<vector::PrintOp>(unkLoc, op);
}

LogicalResult MLIRBench::printMemRef(mlir::Value memRef) {
  // Read into a vector and print output
  // We don't want to alloc the whole tensor as a vector,
  // so we pick the inner dimension and iterate through the outer ones.
  auto outputType = dyn_cast_or_null<MemRefType>(memRef.getType());
  assert(outputType && "Unsupported return type");
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
    bool Ignored;
    vectorFloatValue.convert(APFloat::BFloat(), APFloat::rmNearestTiesToEven,
                             &Ignored);

    minusOne = builder.create<arith::ConstantFloatOp>(
        unkLoc, vectorFloatValue, FloatType::getBF16(builder.getContext()));
  } else {
    minusOne = builder.create<arith::ConstantFloatOp>(unkLoc, vectorFloatValue,
                                                      builder.getF32Type());
  }

  // Vector undefined value
  auto indexType = builder.getIndexType();
  auto countAttr = builder.getIntegerAttr(indexType, outerDims[0]);
  auto count = builder.create<arith::ConstantOp>(unkLoc, indexType, countAttr);
  auto zeroAttr = builder.getIntegerAttr(indexType, 0);
  auto zero = builder.create<arith::ConstantOp>(unkLoc, indexType, zeroAttr);
  auto oneAttr = builder.getIntegerAttr(indexType, 1);
  auto one = builder.create<arith::ConstantOp>(unkLoc, indexType, oneAttr);
  auto loop = builder.create<scf::ForOp>(unkLoc, zero, count, one);
  builder.setInsertionPointToStart(loop.getBody());

  // Loop body
  auto beginIdx = loop.getInductionVar();
  auto vector = builder.create<vector::TransferReadOp>(
      unkLoc, vecType, memRef, ValueRange{beginIdx, zero}, minusOne);
  printVector(vector);

  // Back to main
  builder.setInsertionPointAfter(loop);

  // Finally lower to LLVM Dialect
  return success();
}

LogicalResult MLIRBench::finalize() {
  // If we created a main at all...
  // return void and add func to Module
  if (main) {
    builder.create<func::ReturnOp>(unkLoc);
  }

  // Minimal passes to make it work
  // We don't want TPP passes here, as that's the job of tpp-opt
  // The IR here should be free of TPP/XSMM or any TPP extensions
  PassManager passManager(module->getContext());
  applyPassManagerCLOptions(passManager);

  // Bufferization, if needed
  passManager.addNestedPass<func::FuncOp>(createTensorBufferizePass());
  passManager.addNestedPass<func::FuncOp>(vector::createVectorBufferizePass());
  passManager.addNestedPass<func::FuncOp>(createLinalgBufferizePass());

  // Partial Lowering
  passManager.addPass(createConvertTensorToLinalgPass());
  passManager.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  passManager.addPass(arith::createArithExpandOpsPass());
  passManager.addPass(createConvertVectorToSCFPass());
  passManager.addPass(createConvertSCFToCFPass());

  // Lower to LLVM
  passManager.addPass(createConvertVectorToLLVMPass());
  passManager.addPass(createConvertFuncToLLVMPass());
  passManager.addPass(createMemRefToLLVMConversionPass());
  passManager.addNestedPass<func::FuncOp>(createArithToLLVMConversionPass());
  passManager.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  passManager.addPass(createReconcileUnrealizedCastsPass());

  auto result = passManager.run(module);
  if (failed(result)) {
    llvm::errs() << "ERROR: Failed to lower Module to LLVM dialect\n";
    module->dump();
  }

  return result;
}

//----------------------- Helpers & private methods

llvm::StringRef MLIRBench::createGlobal(MemRefType type) {
  // Simple auto increment
  static unsigned order = 0;

  // TODO: Use some random initialiser
  auto floatValue = APFloat(1.0);
  bool Ignored;
  if (type.getElementType().isBF16()) {
    floatValue.convert(APFloat::BFloat(), APFloat::rmNearestTiesToEven,
                       &Ignored);
  }
  // Create global dense memrefs (Module insertion point)
  auto privAttr = builder.getStringAttr("private");

  // We really only support memrefs as arguments for now
  auto memrefTy = dyn_cast_or_null<MemRefType>(type);
  assert(memrefTy && "Unsupported argument type");

  // Auto incremental naming system
  std::string name = "__wrapper_" + std::to_string(order++);

  // For some reason, memref global op needs dense tensor type
  // See: lib/Dialect/MemRef/IR/MemRefOps.cpp :: GlobalOp::verify
  auto tensorType =
      RankedTensorType::get(memrefTy.getShape(), memrefTy.getElementType());
  auto floatInit = mlir::DenseElementsAttr::get(tensorType, floatValue);
  auto floatType = IntegerType::get(builder.getContext(), 64);
  auto alignment = builder.getIntegerAttr(floatType, 128);

  // Create the global object in the Module's region
  auto global = builder.create<memref::GlobalOp>(unkLoc, StringRef(name),
                                                 privAttr, memrefTy, floatInit,
                                                 /*constant=*/false, alignment);

  return global.getName();
}

void MLIRBench::declareGlobalFunctions() {
  // Common function attributes
  auto cifaceAttr = LLVM::LLVMDialect::getEmitCWrapperAttrName();
  auto unitAttr = UnitAttr::get(module->getContext());
  auto visAttr = builder.getStringAttr("sym_visibility");
  auto privAttr = builder.getStringAttr("private");
  // Data types
  auto f64 = builder.getF64Type();
  auto i64 = builder.getIntegerType(64);

  // Alloc
  timer.alloc = func::FuncOp::create(unkLoc, "timer_alloc",
                                     builder.getFunctionType({}, {i64}));
  timer.alloc->setAttr(cifaceAttr, unitAttr);
  timer.alloc->setAttr(visAttr, privAttr);
  module.push_back(timer.alloc);

  // Start
  timer.start = func::FuncOp::create(unkLoc, "timer_start",
                                     builder.getFunctionType({i64}, {}));
  timer.start->setAttr(cifaceAttr, unitAttr);
  timer.start->setAttr(visAttr, privAttr);
  module.push_back(timer.start);

  // Stop
  timer.stop = func::FuncOp::create(unkLoc, "timer_stop",
                                    builder.getFunctionType({i64}, {}));
  timer.stop->setAttr(cifaceAttr, unitAttr);
  timer.stop->setAttr(visAttr, privAttr);
  module.push_back(timer.stop);

  // Average
  timer.average = func::FuncOp::create(unkLoc, "timer_average",
                                       builder.getFunctionType({i64}, {f64}));
  timer.average->setAttr(cifaceAttr, unitAttr);
  timer.average->setAttr(visAttr, privAttr);
  module.push_back(timer.average);

  // Deviation
  timer.deviation = func::FuncOp::create(unkLoc, "timer_deviation",
                                         builder.getFunctionType({i64}, {f64}));
  timer.deviation->setAttr(cifaceAttr, unitAttr);
  timer.deviation->setAttr(visAttr, privAttr);
  module.push_back(timer.deviation);
}

MemRefType MLIRBench::getGlobalType(llvm::StringRef name) {
  auto op = module.lookupSymbol<memref::GlobalOp>(name);
  assert(op && "memref::Global not found");
  auto memRefTy = dyn_cast_or_null<MemRefType>(op.getType());
  assert(memRefTy && "memref::Global type not a memref?");
  return memRefTy;
}

Block &MLIRBench::getModuleBlock() {
  return module->getRegions().front().front();
}

LogicalResult MLIRBench::emitError(llvm::Twine desc) {
  return module.emitError(desc);
}
