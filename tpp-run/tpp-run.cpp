//===- tpp-run.cpp - TPP CPU Execution Driver------------------------------===//
//
// Main entry point to a command line utility that executes an MLIR file on the
// CPU by translating MLIR to LLVM IR before JIT-compiling and executing the
// latter. Handles TPP/LIBXSMM include/library paths as well as benchmarking
// modes, with warmups, measurements, output comparison, etc.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Casting.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/ExecutionEngine/JitRunner.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

// This is a hack, to parse the command-line options locally
// FIXME: Find a way to grab the options from the JitRunner
struct CmdLineOpts {
  std::string mainFuncName;
} options;

static void locaCmdLineOptParsing(int argc, char **argv) {
  bool nextIsMain = false;
  for (int i=0; i<argc; i++) {
    if (strncmp("-e", argv[i], 2) == 0)
      nextIsMain = true;
    else if (nextIsMain) {
      options.mainFuncName = argv[i];
      break;
    }
  }
}

// Lowers to LLVM Dialect
static LogicalResult lowerToLLVMDialect(ModuleOp module) {
  // Minimal passes to make it work
  // We don't want TPP passes here, as that's the job of tpp-opt
  // The IR here should be free of TPP/XSMM or any TPP extensions
  PassManager passManager(module.getContext());
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
    llvm::errs() << "ERROR: Failed to lower module to LLVM dialect\n";
    module.dump();
  }

  return result;
}

// This function will be called by the pass manager after parsing,
// so we can modify the IR with the needed wrappers
static LogicalResult prepareMLIRKernel(Operation *op) {
  auto module = dyn_cast<ModuleOp>(op);
  if (!module)
    return op->emitOpError("expected a 'builtin.module' op");

  // Find the kernel function and its arguments
  auto moduleRegions = module->getRegions();
  auto& moduleBlock = moduleRegions.front().front();
  auto& moduleOps = moduleBlock.getOperations();

  // If the module is already in the LLVM dialect, recomment mlir-cpu-runner
  for (auto& op: moduleOps) {
    LLVM::LLVMFuncOp llvmFunc = dyn_cast_or_null<LLVM::LLVMFuncOp>(op);
    if (llvmFunc)
      return module.emitError("Module in LLVM Dialect already, use mlir-cpu-runner");
  }

  // The kernel method
  func::FuncOp kernel;

  // If the user passed the entry point, use it
  if (!options.mainFuncName.empty()) {
    for (auto& op: moduleOps) {
      func::FuncOp func = dyn_cast_or_null<func::FuncOp>(op);
      if (!func)
        continue;
      if (func.getName().equals(options.mainFuncName)) {
        kernel = func;
        break;
      }
    }
    if (!kernel)
      return module.emitError("Entry point " + options.mainFuncName + " not found");

  // Else, and there is only one function, use it
  } else if (moduleOps.size() == 1) {
    kernel = dyn_cast_or_null<func::FuncOp>(moduleOps.front());
    if (!kernel)
      return module.emitError("Entry point not in LLVM Dialect");
    options.mainFuncName = kernel.getName();

  // If there is no entry function, and multiple functions, bail
  } else {
    return module.emitError("No valid entry point, use mlir-cpu-runner");
  }

  // If the function has no args or return values, just run it as is
  auto funcType = kernel.getFunctionType();
  if (funcType.getNumInputs() == 0 && funcType.getNumResults() == 0) {
    module.emitRemark("Entry point already created, just running the IR");
    return lowerToLLVMDialect(module);
  }

  // Also ignore functions that return more than one result
  if (funcType.getNumResults() > 1)
    return module.emitError("Multiple return values, use mlir-cpu-runner");

  // Gets a builder on the module
  auto* ctx = module.getContext();
  ctx->getOrLoadDialect<tensor::TensorDialect>();
  ctx->getOrLoadDialect<vector::VectorDialect>();
  OpBuilder builder(ctx);
  Location loc = builder.getUnknownLoc();

  // Rename the entry point to something else and make the main the entry point
  // This is required because we can't change the original options.mainFuncName
  auto name = kernel.getName();
  auto newName = builder.getStringAttr("_" + name);
  kernel.setName(newName);

  // Add a `main` function (with no args/rets) to handle init/tear down
  auto newFuncType = builder.getFunctionType({}, {});
  auto main = func::FuncOp::create(loc, name, newFuncType);
  main.setVisibility(SymbolTable::Visibility::Public);
  auto entryBlock = main.addEntryBlock();

  // Initialise the inputs as global constants
  // TODO: Use some random initialiser
  APFloat floatValue = APFloat(1.0F);

  // Create global dense memrefs (module insertion point)
  builder.setInsertionPointToStart(&moduleBlock);
  auto privAttr = builder.getStringAttr("private");
  int order = 0;
  for (auto& ty: funcType.getInputs()) {
    // We really only support memrefs as arguments for now
    auto memrefTy = dyn_cast_or_null<MemRefType>(ty);
    assert(memrefTy && "Unsupported argument type");

    // Global op properties
    std::string name = "__wrapper_" + std::to_string(order++);
    // For some reason, memref global op needs dense tensor type
    // See: lib/Dialect/MemRef/IR/MemRefOps.cpp :: GlobalOp::verify
    auto tensorType = RankedTensorType::get(memrefTy.getShape(), memrefTy.getElementType());
    auto floatInit = mlir::DenseElementsAttr::get(tensorType, floatValue);
    auto alignment = builder.getIntegerAttr(builder.getI64Type(), 128);

    // Create the global object in the module's region
    builder.create<memref::GlobalOp>(loc, StringRef(name), privAttr, memrefTy, floatInit, /*constant=*/false, alignment);
  }

  // Get those globals as arguments (function insertion point)
  builder.setInsertionPointToStart(entryBlock);
  SmallVector<Value> args;
  order = 0;
  for (auto& ty: funcType.getInputs()) {
    // GetGlobal op properties
    std::string name = "__wrapper_" + std::to_string(order++);
    auto nameAttr = builder.getStringAttr(name);
    auto getGlobal = builder.create<memref::GetGlobalOp>(loc, ty, nameAttr);

    // Add argument to list
    args.push_back(getGlobal);
  }

  // Call the kernel
  Value result;
  if (funcType.getNumResults() == 0) {
    builder.create<func::CallOp>(loc, kernel, args);
    result = args.back();
  } else {
    auto call = builder.create<func::CallOp>(loc, kernel, args);
    result = call->getOpResult(0);
  }

  // Read into a vector and print output
  // We don't want to alloc the whole tensor as a vector,
  // so we pick the inner dimension and iterate through the outer ones.
  auto outputType = dyn_cast_or_null<MemRefType>(result.getType());
  assert(outputType && "Unsupported return type");
  VectorType vecType;
  auto lastDim = outputType.getRank() - 1;
  ArrayRef<int64_t> outer_dims(1);
  if (outputType.getRank() > 1) {
    ArrayRef<int64_t> inner_dims(&outputType.getShape()[lastDim], 1);
    vecType = VectorType::get(inner_dims, outputType.getElementType());
    outer_dims = ArrayRef<int64_t>(&outputType.getShape()[0], outputType.getRank()-1);
  } else {
    vecType = VectorType::get(outputType.getShape(), outputType.getElementType());
  }

  APFloat vectorFloatValue = APFloat(-1.0F);
  auto minusOne = builder.create<arith::ConstantFloatOp>(loc, vectorFloatValue, builder.getF32Type());
  auto zeroIdx = builder.create<arith::ConstantIndexOp>(loc, 0);
  auto indices = ValueRange{zeroIdx, zeroIdx};
  // TODO: Create a loop in IR
  assert(outer_dims.size() == 1 && "Only supports 2D tensors for now");
  for (int i=0; i<outer_dims[0]; i++) {
    auto vector = builder.create<vector::TransferReadOp>(loc, vecType, result, indices, minusOne);
    builder.create<vector::PrintOp>(loc, vector);
  }

  // Return void and add func to module
  builder.create<func::ReturnOp>(loc);
  module.push_back(main);

  // Finally lower to LLVM Dialect
  return lowerToLLVMDialect(module);
}

// This function will be called at the end, to emit an LLVM Module.
// It may not be necessary, but here just in case.
static std::unique_ptr<llvm::Module> buildLLVMModule(Operation *op, llvm::LLVMContext& context) {
  auto module = dyn_cast<ModuleOp>(op);
  assert(module && "expected a 'builtin.module' op");

  // FIXME: This should detect library paths for both MLIR and TPP libraries
  std::unique_ptr<llvm::Module> llvm = translateModuleToLLVMIR(module, context);
  return llvm;
}

int main(int argc, char **argv) {
  // Initialize the LLVM machinery
  llvm::InitLLVM y(argc, argv);
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  DialectRegistry registry;
  registerAllDialects(registry);
  registerAllToLLVMIRTranslations(registry);

  // This is how we integrate with the pipeline
  JitRunnerConfig config;
  config.mlirTransformer = prepareMLIRKernel;
  config.llvmModuleBuilder = buildLLVMModule;

  // Hack
  locaCmdLineOptParsing(argc, argv);

  // Call the main JIT function
  return JitRunnerMain(argc, argv, registry, config);
}
