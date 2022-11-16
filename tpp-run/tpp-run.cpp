//===- tpp-run.cpp - TPP CPU Execution Driver------------------------------===//
//
// Main entry point to a command line utility that executes an MLIR file on the
// CPU by translating MLIR to LLVM IR before JIT-compiling and executing the
// latter. Handles TPP/LIBXSMM include/library paths as well as benchmarking
// modes, with warmups, measurements, output comparison, etc.
//
//===----------------------------------------------------------------------===//

#include "MLIRBench.h"

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
#include "mlir/ExecutionEngine/JitRunner.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

using namespace mlir;

// This function will be called by the pass manager after parsing,
// so we can modify the IR with the needed wrappers
static LogicalResult prepareMLIRKernel(Operation *Op,
                                       JitRunnerOptions &Options) {
  MLIRBench Bench(Op);

  if (Options.mainFuncType != "void")
    return Bench.emitError(
        "Main function has to be 'void', even if the kernel return's a value, "
        "because that's the type of the wrapper we create here");

  if (failed(Bench.findKernel(Options.mainFuncName)))
    return Bench.emitError("Cannot find kernel '" + Options.mainFuncName + "'");

  if (failed(Bench.checkKernelSignature()))
    return Bench.finalize();

  if (failed(Bench.renameKernel()))
    return Bench.emitError("Cannot rename kernel function");

  if (failed(Bench.createMainWrapper()))
    return Bench.emitError("Cannot create main wrapper");

  SmallVector<llvm::StringRef> GlobalList;
  if (failed(Bench.createGlobals(GlobalList)))
    return Bench.emitError("Cannot create the global memrefs");

  // TODO: Insert the benchmark loop here
  auto Return = Bench.callKernel(GlobalList);
  if (!Return)
    return Bench.emitError("Cannot generate a call to the kernel");

  // TODO: Insert the statistics here

  // TODO: We may not want to print on benchmark runs...
  if (failed(Bench.printMemRef(Return)))
    return Bench.emitError("Cannot print result memref");

  // Finally lower to LLVM Dialect
  return Bench.finalize();
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

  // Call the main JIT function
  return JitRunnerMain(argc, argv, registry, config);
}
