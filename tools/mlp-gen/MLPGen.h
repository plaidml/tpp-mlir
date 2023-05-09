//===- MLPGen.h MLP Generator ---------------------------------------------===//
//
// Class that handles MLIR generation for the MLP options.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include "TPP/BuilderUtils.h"

#include <optional>

namespace mlir {
class ModuleOp;
class MemRefType;
class Operation;
class Value;
namespace func {
class FuncOp;
} // namespace func

/// MLP Generator: produces MLIR linalg-on-tensor dialect for an MLP mopdel with
/// the appropriate number of hidden layers and other properties selected.
class MLPGenerator {
  /// MLIR Context
  MLIRContext context;

  /// MLIR OpBulder
  OpBuilder builder;

  /// Unknown location, since all this code is auto-generated
  Location loc;

  /// Main module
  ModuleOp module;

  /// Mini-Batch
  unsigned miniBatch;

  /// Layer sizes
  SmallVector<int64_t> layers;

  /// Tile sizes
  SmallVector<int64_t> tiles;

  /// Data type (element type of all tensors)
  Type dataType;

  /// Random seed
  int seed;

  /// Tensor init type
  TensorInitType initType;

  // ============================ Code Generation Options

  /// Lower softmax at the last layer
  bool enableSoftmax;

  /// Initialize accumulation matrix with bias
  bool biasAcc;

  /// List of supported kernel types that can be generated
  enum class KernelType { MLP, MATMUL, FULLY_CONNECTED };

  /// Type of kernel to be generated
  KernelType kernelType;

  // ============================ Helpers

  /// Return current random seed, update next
  int getRand();

  /// Type of packing (NxC, KxC, NxK)
  enum PackingType { PACK_INPUT, PACK_WEIGHT, PACK_OUTPUT };

  /// Return shaped type (packed if requested)
  TensorType getShape(ArrayRef<int64_t>, PackingType);

  /// Affine expressions for maps
  SmallVector<AffineExpr, 6> affineExprs;

  enum MapType {
    MAP_PARALLEL,
    MAP_REDUCTION,
    MAP_MATMUL_INPUT,
    MAP_MATMUL_WEIGHT,
    MAP_MATMUL_OUTPUT,
    MAP_MATMUL // Alias for iterator type
  };

  /// Return affine map (packed if requested)
  /// If order is not empty, re-order the dims in that order
  /// If dims is passed, force number of dims, otherwise, take from tensor
  /// If reduction is true, emit zeroExpr for the tail reduction
  AffineMap getMap(Value, MapType);

  /// Return the iterator types for a particular map type
  /// Add iterators if the types are packed
  SmallVector<utils::IteratorType> getIterators(MapType);

  // ============================ Core Logic
  // To avoid allocating new tensors, we bind the output of the matmul to the
  // input of the bias add, make it in-place and bind that to the input of
  // the ReLU, also making it in-place, and returning the first alloc.

  /// Some arguments are optional, so we use this struct to simplify the
  /// argument handling in lowerMatmul
  struct MatMulArgs {
    Value input;
    Value weight;
    Value bias;
    Value output;
  };

  /// Creates a matmul in the current function
  Value lowerMatmul(MatMulArgs);

  /// Creates a bias add in the current function
  Value lowerBiasAdd(Value, Value);

  /// Creates a relu in the current function
  Value lowerRelu(Value);

  /// Creates a softmax in the current function
  Value lowerSoftmax(Value, Value);

  // ============================ Main API

  /// Creates a hidden layer function, to be called by the kernel
  /// There will be one per hidden layer
  Value createLayer(unsigned, Value);

  /// Creates an output layer function, to be called by the kernel
  /// Classifies the output of the last layer and put it in the second argumnent
  Value createOutputLayer(Value, Value);

  /// Creates an MLP kernel
  void createMlpKernel();

  /// Creates a Matmul kernel
  void createMatmulKernel();

  /// Creates the entry point, that creates and executes chosen kernel
  /// No need to return the function, as all we need is to dump the module
  void createEntryPoint();

public:
  /// Creates a specific module. Different configurations need different modules
  /// so should create new objects to not have to share / cleanup existing MLIR
  /// modules.
  MLPGenerator(StringRef, unsigned, StringRef, StringRef, unsigned, int, bool,
               bool);

  ~MLPGenerator() { module->destroy(); }

  /// Generates the whole IR and write to file
  /// Return 0 on success, 1 on failure
  int generate(StringRef filename);
};

} // namespace mlir
