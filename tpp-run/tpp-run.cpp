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

#include "TPP/Dialect/Check/BufferizableOpInterfaceImpl.h"
#include "TPP/Dialect/Check/CheckDialect.h"
#include "TPP/Dialect/LinalgX/BufferizableOpInterfaceImpl.h"
#include "TPP/Dialect/LinalgX/LinalgXDialect.h"
#include "TPP/Dialect/Perf/BufferizableOpInterfaceImpl.h"
#include "TPP/Dialect/Perf/PerfDialect.h"
#include "TPP/Dialect/Tpp/TppDialect.h"
#include "TPP/Dialect/Transform/LinalgXTransformOps.h"
#include "TPP/Dialect/VNNI/BufferizableOpInterfaceImpl.h"
#include "TPP/Dialect/VNNI/VNNIDialect.h"
#include "TPP/Dialect/Xsmm/XsmmDialect.h"
#include "TPP/Passes.h"

using namespace mlir;

// Number of loops for benchmarks
llvm::cl::opt<unsigned>
    benchNumLoops("n", llvm::cl::desc("Number of loops for benchmarks"),
                  llvm::cl::value_desc("int"), llvm::cl::init(1));

// Print result
llvm::cl::opt<bool> printKernelResult("print",
                                      llvm::cl::desc("Print kernel result"),
                                      llvm::cl::value_desc("true/false"),
                                      llvm::cl::init(false));

// This function will be called by the pass manager after parsing,
// so we can modify the IR with the needed wrappers
static LogicalResult prepareMLIRKernel(Operation *op,
                                       JitRunnerOptions &options) {
  MLIRBench bench(op);

  // Basic checks
  if (options.mainFuncType != "void")
    return bench.emitError(
        "Main function has to be 'void', even if the kernel return's a value, "
        "because that's the type of the wrapper we create here");

  if (failed(bench.findKernel(options.mainFuncName)))
    return bench.emitError("Cannot find kernel '" + options.mainFuncName + "'");

  if (failed(bench.checkKernelSignature()))
    return bench.finalize();

  // Move the kernel to a local name, so we can create `main` with the same
  // name as the pre-defined entry point (since we can't change it)
  if (failed(bench.renameKernel()))
    return bench.emitError("Cannot rename kernel function");

  // Creates the main wrapper
  if (failed(bench.createMainWrapper()))
    return bench.emitError("Cannot create main wrapper");

  // Creates the inputs for the kernel
  if (failed(bench.createKernelArgs()))
    return bench.emitError("Cannot create kernel inputs");

  // Call kernel once, to bootstrap (JIT compile, warm up caches)
  auto call = bench.callKernel();
  if (!call)
    return bench.emitError("Cannot generate a call to the kernel");

  // Print the result of the warming up, should be the same as any other
  if (printKernelResult && failed(bench.printResult(call)))
    return bench.emitError("Cannot print result memref");

  // This is the main loop, if N > 1
  if (benchNumLoops > 1) {
    auto acc = bench.createTimerLoop(benchNumLoops);
    if (!acc)
      return bench.emitError("Cannot create timer loop");
    auto stats = bench.getTimerStats(acc);
    if (!stats)
      return bench.emitError("Cannot get timer stats");
    bench.printVector(stats);
  }

  // Finally lower to LLVM Dialect
  return bench.finalize();
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
  registry.insert<mlir::tpp::TppDialect>();
  registry.insert<mlir::xsmm::XsmmDialect>();
  registry.insert<mlir::linalgx::LinalgXDialect>();
  registry.insert<mlir::check::CheckDialect>();
  registry.insert<mlir::vnni::VNNIDialect>();
  registry.insert<mlir::perf::PerfDialect>();
  mlir::linalgx::registerTransformDialectExtension(registry);
  mlir::linalgx::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::check::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::vnni::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::perf::registerBufferizableOpInterfaceExternalModels(registry);
  registerAllDialects(registry);
  registerAllToLLVMIRTranslations(registry);

  // This is how we integrate with the pipeline
  JitRunnerConfig config;
  config.mlirTransformer = prepareMLIRKernel;

  // Call the main JIT function
  return JitRunnerMain(argc, argv, registry, config);
}
