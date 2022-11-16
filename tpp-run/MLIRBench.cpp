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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

// Number of loops for benchmarks
// llvm::cl::opt<unsigned>
//    numLoops("n", llvm::cl::desc("Number of loops for benchmarks"),
//             llvm::cl::value_desc("int"), llvm::cl::init(1));

MLIRBench::MLIRBench(mlir::Operation *Op)
    : Builder(Op->getContext()), UnkLoc(Builder.getUnknownLoc()) {
  Module = dyn_cast<ModuleOp>(Op);
  assert(Module && "expected a 'builtin.Module' op");
  auto *Ctx = Module->getContext();
  Ctx->getOrLoadDialect<tensor::TensorDialect>();
  Ctx->getOrLoadDialect<vector::VectorDialect>();
}

LogicalResult MLIRBench::findKernel(StringRef Name) {
  auto &ModuleOps = getModuleBlock().getOperations();
  if (!Name.empty()) {
    // If the user passed the entry point, use it
    for (auto &Op : ModuleOps) {
      func::FuncOp Func = dyn_cast_or_null<func::FuncOp>(Op);
      if (Func && Func.getName().equals(Name)) {
        Kernel = Func;
        break;
      }
    };
    // Needs to be in the func dialect, not LLVM
    if (!Kernel)
      return Module.emitError("Entry point " + Name +
                              " not found as a func.func");

  } else if (ModuleOps.size() == 1) {
    // Else, and there is only one function, use it
    Kernel = dyn_cast_or_null<func::FuncOp>(ModuleOps.front());
    if (!Kernel)
      return Module.emitError("Single function not in Func Dialect");

  } else {
    // If there is no entry function, and multiple functions, bail
    return Module.emitError("No valid entry point, use mlir-cpu-runner");
  }

  // Ignore functions that return more than one result
  auto FuncType = Kernel.getFunctionType();
  if (FuncType.getNumResults() > 1)
    return Module.emitError("Multiple return values, use mlir-cpu-runner");

  return success();
}

LogicalResult MLIRBench::checkKernelSignature() {
  // If the function has no args or return values, just run it as is
  auto FuncType = Kernel.getFunctionType();
  if (FuncType.getNumInputs() == 0 && FuncType.getNumResults() == 0)
    return Module.emitError("Entry point already created, use mlir-cpu-runner");

  return success();
}

LogicalResult MLIRBench::renameKernel() {
  // Rename the entry point to something else and make the main the entry point
  // This is required because we can't change the original Name
  MainName = Kernel.getName();
  auto NewName = Builder.getStringAttr("_" + MainName);
  Kernel.setName(NewName);

  return success();
}

LogicalResult MLIRBench::createMainWrapper() {
  // Add a `main` function (with no args/rets) to handle init/tear down
  auto FuncType = Builder.getFunctionType({}, {});
  Main = func::FuncOp::create(UnkLoc, MainName, FuncType);
  Main.setVisibility(SymbolTable::Visibility::Public);
  Main.addEntryBlock();

  return success();
}

LogicalResult
MLIRBench::createGlobals(llvm::SmallVector<llvm::StringRef> &List) {
  // Create global dense memrefs (Module insertion point)
  Builder.setInsertionPointToStart(&getModuleBlock());
  auto FuncType = Kernel.getFunctionType();
  for (auto &Ty : FuncType.getInputs()) {
    auto MemRefTy = dyn_cast_or_null<MemRefType>(Ty);
    List.push_back(createGlobal(MemRefTy));
  }

  return success();
}

Value MLIRBench::callKernel(llvm::SmallVector<llvm::StringRef> &List) {
  // Get those globals as arguments (function insertion point)
  Builder.setInsertionPointToEnd(&Main.getBlocks().front());
  SmallVector<Value> Args;
  for (auto &Name : List) {
    // GetGlobal op properties
    auto NameAttr = Builder.getStringAttr(Name);
    auto Type = getGlobalType(Name);
    auto GetGlobal =
        Builder.create<memref::GetGlobalOp>(UnkLoc, Type, NameAttr);

    // Add argument to list
    Args.push_back(GetGlobal);
  }

  // Call the Kernel, making sure to set the result to either the return value
  // or the last argument, if the return is void.
  Value Result;
  auto FuncType = Main.getFunctionType();
  if (FuncType.getNumResults() == 0) {
    Builder.create<func::CallOp>(UnkLoc, Kernel, Args);
    Result = Args.back();
  } else {
    auto Call = Builder.create<func::CallOp>(UnkLoc, Kernel, Args);
    Result = Call->getOpResult(0);
  }

  return Result;
}

LogicalResult MLIRBench::printMemRef(mlir::Value MemRef) {
  // Read into a vector and print output
  // We don't want to alloc the whole tensor as a vector,
  // so we pick the inner dimension and iterate through the outer ones.
  auto outputType = dyn_cast_or_null<MemRefType>(MemRef.getType());
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

  APFloat vectorFloatValue = APFloat(-1.0F);
  auto minusOne = Builder.create<arith::ConstantFloatOp>(
      UnkLoc, vectorFloatValue, Builder.getF32Type());
  // TODO: Create a loop in IR
  auto zeroIdx = Builder.create<arith::ConstantIndexOp>(UnkLoc, 0);
  assert(outerDims.size() == 1 && "Only supports 2D tensors for now");
  for (int i = 0; i < outerDims[0]; i++) {
    auto beginIdx = Builder.create<arith::ConstantIndexOp>(UnkLoc, i);

    auto indices = ValueRange{beginIdx, zeroIdx};
    auto vector = Builder.create<vector::TransferReadOp>(
        UnkLoc, vecType, MemRef, indices, minusOne);
    Builder.create<vector::PrintOp>(UnkLoc, vector);
  }

  // Finally lower to LLVM Dialect
  return success();
}

LogicalResult MLIRBench::finalize() {
  // If we created a main at all...
  // return void and add func to Module
  if (Main) {
    Builder.create<func::ReturnOp>(UnkLoc);
    Module.push_back(Main);
  }

  // Minimal passes to make it work
  // We don't want TPP passes here, as that's the job of tpp-opt
  // The IR here should be free of TPP/XSMM or any TPP extensions
  PassManager PassManager(Module->getContext());
  applyPassManagerCLOptions(PassManager);

  // Bufferization, if needed
  PassManager.addNestedPass<func::FuncOp>(createTensorBufferizePass());
  PassManager.addNestedPass<func::FuncOp>(vector::createVectorBufferizePass());
  PassManager.addNestedPass<func::FuncOp>(createLinalgBufferizePass());

  // Partial Lowering
  PassManager.addPass(createConvertTensorToLinalgPass());
  PassManager.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  PassManager.addPass(arith::createArithExpandOpsPass());
  PassManager.addPass(createConvertVectorToSCFPass());
  PassManager.addPass(createConvertSCFToCFPass());

  // Lower to LLVM
  PassManager.addPass(createConvertVectorToLLVMPass());
  PassManager.addPass(createConvertFuncToLLVMPass());
  PassManager.addPass(createMemRefToLLVMConversionPass());
  PassManager.addNestedPass<func::FuncOp>(createArithToLLVMConversionPass());
  PassManager.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  PassManager.addPass(createReconcileUnrealizedCastsPass());

  auto Result = PassManager.run(Module);
  if (failed(Result)) {
    llvm::errs() << "ERROR: Failed to lower Module to LLVM dialect\n";
    Module->dump();
  }

  return Result;
}

//----------------------- Helpers & private methods

llvm::StringRef MLIRBench::createGlobal(MemRefType Type) {
  // Simple auto increment
  static unsigned Order = 0;

  // TODO: Use some random initialiser
  APFloat FloatValue = APFloat(1.0F);

  // Create global dense memrefs (Module insertion point)
  auto PrivAttr = Builder.getStringAttr("private");

  // We really only support memrefs as arguments for now
  auto MemrefTy = dyn_cast_or_null<MemRefType>(Type);
  assert(MemrefTy && "Unsupported argument type");

  // Auto incremental naming system
  std::string Name = "__wrapper_" + std::to_string(Order++);

  // For some reason, memref global op needs dense tensor type
  // See: lib/Dialect/MemRef/IR/MemRefOps.cpp :: GlobalOp::verify
  auto TensorType =
      RankedTensorType::get(MemrefTy.getShape(), MemrefTy.getElementType());
  auto FloatInit = mlir::DenseElementsAttr::get(TensorType, FloatValue);
  auto Alignment = Builder.getIntegerAttr(Builder.getI64Type(), 128);

  // Create the global object in the Module's region
  auto Global = Builder.create<memref::GlobalOp>(UnkLoc, StringRef(Name),
                                                 PrivAttr, MemrefTy, FloatInit,
                                                 /*constant=*/false, Alignment);

  return Global.getName();
}

MemRefType MLIRBench::getGlobalType(llvm::StringRef Name) {
  auto Op = Module.lookupSymbol<memref::GlobalOp>(Name);
  assert(Op && "memref::Global not found");
  auto MemRefTy = dyn_cast_or_null<MemRefType>(Op.getType());
  assert(MemRefTy && "memref::Global type not a memref?");
  return MemRefTy;
}

Block &MLIRBench::getModuleBlock() {
  return Module->getRegions().front().front();
}

LogicalResult MLIRBench::emitError(llvm::Twine Desc) {
  return Module.emitError(Desc);
}
