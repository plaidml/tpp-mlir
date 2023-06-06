//===- TensorInitInt.h - MLIR Tensor Initialization -----------------------===//
//
// Initializes tensors for kernel input/output handling with some reasonable
// distribution to allow for layout testing (reorder, pad) without vanishing
// or exploding values at the end of a large model (0.0 ~ 1.0).
//
//===----------------------------------------------------------------------===//

#ifndef TPP_TENSORINITINT_H
#define TPP_TENSORINITINT_H

#include "TPP/TensorInit.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"

#include <algorithm>
#include <random>

// Base class for integer values.
struct TensorInitInt : public TensorInit<llvm::APInt> {
  // Data type. (TODO: Support more data types)
  enum class DataType { I8, I16, I32, I64 };

  static bool isTypeSupported(const mlir::Type &type) {
    return type.isInteger(8) || type.isInteger(16) || type.isInteger(32) ||
           type.isInteger(64);
  }

  // Get data type from element type.
  static DataType getTensorInitDataType(mlir::Type type);

  // Get bit width from data type.
  static unsigned getDataTypeBitWidth(DataType type);

  // True if the data type is signed.
  static bool isDataTypeSigned(DataType type);

  TensorInitInt(DataType type)
      : type(type), bitWidth(getDataTypeBitWidth(type)),
        isSigned(isDataTypeSigned(type)) {}
  virtual ~TensorInitInt() = default;

protected:
  // Tensor element data type.
  DataType type;

  // Bit width of the data type.
  unsigned bitWidth;

  // True if the data type is signed.
  bool isSigned;

  // Insert element indexed on the buffer.
  using TensorInit::insert;
  void insert(size_t index, uint64_t value);

  // Insert element at the end of the buffer.
  using TensorInit::push;
  void push(uint64_t value);

  // Convert value to the tensor's data type (by reference).
  void convertType(llvm::APInt &value) override final;

  // Actual implementation that fills the buffer
  // To be implemented by derived classes.
  virtual void fillData() override = 0;
};

// Constant init (all-ones, do not use!).
struct ConstantTensorInitInt : TensorInitInt {
  ConstantTensorInitInt(DataType type) : TensorInitInt(type) {}

  // Return a dense<1> repeated throughout the shape.
  mlir::DenseElementsAttr get(mlir::ShapedType shape) override;

  void fillData() override;
};

// Simple init (basic example, not useful).
struct SimpleTensorInitInt : TensorInitInt {
  SimpleTensorInitInt(DataType type) : TensorInitInt(type) {}

  // Return a dense<0, 1, 2> repeated throughout the shape.
  void fillData() override;
};

// Continuous init (quantized normalized affine range).
struct ContinuousTensorInitInt : TensorInitInt {
  ContinuousTensorInitInt(DataType type) : TensorInitInt(type) {}

  // Return a dense<0 ... upperBound> throughout the shape.
  void fillData() override;

  // Upper bound for quantization.
  int upperBound = 255;
};

// Random init (uniform).
struct RandomTensorInitInt : TensorInitInt {
  RandomTensorInitInt(DataType type, int seed)
      : TensorInitInt(type), generator(seed), distribution(0, 255) {}

  // Next random uniform number.
  float next() { return distribution(generator); }

  // Return a dense<uniform(0, 255)> throughout the shape.
  void fillData() override;

private:
  // Random generator.
  std::default_random_engine generator;
  // Random distribution.
  std::uniform_int_distribution<uint64_t> distribution;
};

// Random init (normal).
struct NormalTensorInitInt : TensorInitInt {
  NormalTensorInitInt(DataType type, int seed)
      : TensorInitInt(type), generator(seed), distribution(255, 0.5) {}

  // Next random number.
  float next() {
    auto value = distribution(generator);
    return value;
  }

  // Return a dense<normal(0.0, 1.0)> throughout the shape.
  void fillData() override;

private:
  // Random generator.
  std::default_random_engine generator;
  // Random distribution.
  std::binomial_distribution<uint64_t> distribution;
};

#endif // TPP_TENSORINITINT_H
