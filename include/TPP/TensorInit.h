#ifndef TPP_RUN_TENSORINIT_H
#define TPP_RUN_TENSORINIT_H

//===- TensorInit.h - MLIR Tensor Initialization --------------------------===//
//
// Initializes tensors for kernel input/output handling with some reasonable
// distribution to allow for layout testing (reorder, pad) without vanishing
// or exploding values at the end of a large model (0.0 ~ 1.0).
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"

#include <algorithm>
#include <random>
#include <vector>

/// Base class
/// Assumes float (32) as base type.
/// TODO: Add a template parameter if/when we support double.
struct TensorInit {
  /// Data type (TODO: Support 64/8-bit data types)
  enum DataType {
    INVALID, FP32, BF16
  };

  /// Get the data type for the float with width bits
  static DataType getFloatDataType(unsigned widthInBits) {
    switch (widthInBits) {
      case 16:
        return BF16;
      case 32:
        return FP32;
      default:
        return INVALID;
    }
  }

protected:
  /// BF16 conversion (by reference)
  static void toBF16(llvm::APFloat& value) {
    bool ignored;
    value.convert(llvm::APFloat::BFloat(), llvm::APFloat::rmNearestTiesToEven,
                       &ignored);
  }

  /// Data type
  DataType type;
  /// Number of elements in the shape
  size_t size;
  /// Data pointer
  std::vector<llvm::APFloat> buffer;

  /// Resize the buffer with the total allocation size
  void allocateBuffer();

  /// Insert element indexed on the buffer
  void insert(size_t index, float value);

  /// Insert element at the end of the buffer
  void push_back(float value);

  /// Actual implementation that fills the buffer
  /// To be implemented by derived classes.
  virtual void fillData() = 0;

public:
  /// Returns a dense attribute with a specified shape, initialized
  /// with a particular implementation (see derived classes) with
  /// a reasonable distribution (0.0 ~ 1.0)
  virtual mlir::DenseElementsAttr get(mlir::ShapedType shape);

  /// DEBUG ONLY: Print a specific value as an fp32 (regardless of data type)
  float at(size_t index);

  TensorInit(DataType type) : type(type), size(1) {}
  virtual ~TensorInit() {}
};

/// Constant init (all-ones, do not use!)
struct ConstantTensorInit : TensorInit {
  ConstantTensorInit(DataType type) : TensorInit(type) {}

  /// Return a dense<1.0> repeated throughout the shape
  mlir::DenseElementsAttr get(mlir::ShapedType shape) override;

  void fillData() override;
};

/// Simple init (basic example, not useful)
struct SimpleTensorInit : TensorInit {
  SimpleTensorInit(DataType type) : TensorInit(type) {}

  /// Return a dense<0.3, 0.6, 0.9> repeated throughout the shape
  void fillData() override;
};

/// Continuous init (normalized affine range)
struct ContinuousTensorInit : TensorInit {
  ContinuousTensorInit(DataType type) : TensorInit(type) {}

  /// Return a dense<0.0 ... 1.0> throughout the shape
  void fillData() override;
};

/// Random init (uniform)
struct RandomTensorInit : TensorInit {
  RandomTensorInit(DataType type, int seed)
      : TensorInit(type), generator(seed), distribution(0.0, 1.0) {}

  /// Random generator
  std::default_random_engine generator;
  /// Random distribution
  std::uniform_real_distribution<float> distribution;

  /// Next random uniform number
  float next() {
    return distribution(generator);
  }

  /// Return a dense<uniform(0.0, 1.0)> throughout the shape
  void fillData() override;
};

/// Random init (normal)
struct NormalTensorInit : TensorInit {
  NormalTensorInit(DataType type, int seed)
      : TensorInit(type), generator(seed), distribution(0.0, 0.2) {}

  /// Random generator
  std::default_random_engine generator;
  /// Random distribution
  std::normal_distribution<float> distribution;

  /// Next random number
  float next() {
    auto value = distribution(generator);
    value = std::clamp(value, 0.0f, 1.0f);
    return value;
  }

  /// Return a dense<normal(0.0, 1.0)> throughout the shape
  void fillData() override;
};

#endif
