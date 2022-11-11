//===- MLIRBench.h - MLIR Benchmark Producer ------------------------------===//
//
// Producer for benchmark wrapper methods. Upon selecting a kernel to run, maps
// the arguments, random initialize them and call the kernel as many times as
// requested, taking measurements and printing the result in the end.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
  class ModuleOp;
  class MemRefType;
  class Operation;
  class Value;
  namespace func {
    class FuncOp;
  } // namespace func

/// MLIRBench - Creates wrapper for calling kernel methods.
///
/// Note: This class is a mix between a utility class and a driver
/// because I still don't know which way we want to go. For now, the
/// inteface is a bit weird, but it will get better once we clear the
/// API design, with time.
class MLIRBench {
  /// MLIR OpBulder
  mlir::OpBuilder Builder;

  /// Unknown location, since all this code is auto-generated
  mlir::Location UnkLoc;

  /// Main module
  mlir::ModuleOp Module;

  /// Kernel function, if found
  mlir::func::FuncOp Kernel;

  /// Main wrapper function, calls kernel
  mlir::func::FuncOp Main;

  /// Local cache of the main name
  llvm::StringRef MainName;

  /// Global variables for all arguments (in order)
  llvm::SmallVector<llvm::StringRef> Globals;

  /// Create a random global based on the memref type
  llvm::StringRef createGlobal(mlir::MemRefType Type);

  /// Create a random global based on the memref type
  mlir::MemRefType getGlobalType(llvm::StringRef Name);

  /// Gets module's main block
  mlir::Block& getModuleBlock();

public:
  /// Creates context, builder
  MLIRBench(mlir::Operation* Op);

  /// Finds the kernel method, checks correct name and shape
  mlir::LogicalResult findKernel(llvm::StringRef Name);

  /// Check if the kernel is already an entry point
  /// Find the kernel first with findKernel.
  mlir::LogicalResult checkKernelSignature();

  /// Renames the kernel to _name, so that we can create the wrapper
  mlir::LogicalResult renameKernel();

  /// Create main wrapper function, sets insertion point
  mlir::LogicalResult createMainWrapper();

  /// Create all globals for the kernel method initializers
  /// Populates the list with the names, in order
  mlir::LogicalResult createGlobals(llvm::SmallVector<llvm::StringRef>& List);

  /// Calls the kernel, returns the result, which is either
  /// the return value (if any) or the last argument (outs).
  mlir::Value callKernel(llvm::SmallVector<llvm::StringRef>& List);

  /// Prints the memref as a vector read + print
  mlir::LogicalResult printMemRef(mlir::Value MemRef);

  /// Terminates the function, issuing a return, lower to LLVM
  mlir::LogicalResult finalize();

  /// Reports error on the current module's location
  mlir::LogicalResult emitError(llvm::Twine Desc);
};

} // namespace mlir
