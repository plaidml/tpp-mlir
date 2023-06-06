//===- TensorInit.h - MLIR Tensor Initialization --------------------------===//
//
// Initializes tensors for kernel input/output handling with some reasonable
// distribution to allow for layout testing (reorder, pad) without vanishing
// or exploding values at the end of a large model (0.0 ~ 1.0).
//
//===----------------------------------------------------------------------===//

#ifndef TPP_TENSORINIT_H
#define TPP_TENSORINIT_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"

#include <vector>

// Interface.
struct ITensorInit {
  ITensorInit() = default;
  virtual ~ITensorInit() = default;

  // Returns a dense attribute with a specified shape, initialized
  // with a particular implementation (see derived classes) with
  // a reasonable distribution.
  virtual mlir::DenseElementsAttr get(mlir::ShapedType shape) = 0;
};

// Base class.
template <typename T> struct TensorInit : public ITensorInit {
  TensorInit() : size(1) {}
  virtual ~TensorInit() = default;

  // Returns a dense attribute with a specified shape, initialized
  // with a particular implementation (see derived classes) with
  // a reasonable distribution.
  virtual mlir::DenseElementsAttr get(mlir::ShapedType shape) override {
    buffer.clear();
    for (size_t dim = 0, rank = shape.getRank(); dim < rank; dim++)
      size *= shape.getDimSize(dim);
    fillData();
    // For some reason, memref global op needs dense tensor type
    // See: lib/Dialect/MemRef/IR/MemRefOps.cpp :: GlobalOp::verify
    auto tensorType =
        mlir::RankedTensorType::get(shape.getShape(), shape.getElementType());
    return mlir::DenseElementsAttr::get(tensorType, buffer);
  }

protected:
  // Number of elements in the shape
  size_t size;
  // Data pointer
  std::vector<T> buffer;

  // Insert element indexed on the buffer
  virtual void insert(size_t index, T value) {
    buffer[index] = value;
    convertType(buffer[index]);
  }

  // Insert element at the end of the buffer
  virtual void push(T value) {
    buffer.push_back(value);
    convertType(buffer.back());
  }

  // Convert value to the tensor's data type (by reference)
  virtual void convertType(T &value) = 0;

  // Actual implementation that fills the buffer
  // To be implemented by derived classes.
  virtual void fillData() = 0;
};

// Initialization type, to use with the getter below
enum class TensorInitType {
  Auto,
  Constant,
  Simple,
  Continuous,
  Random,
  Normal,
  Invalid
};

// Unique pointer for tensor init to help with memory management
using TensorInitPtr = std::unique_ptr<ITensorInit>;

// Parse init type string into TensorInitType
TensorInitType parseTensorInitType(llvm::StringRef name);

// Return an initializer smart pointer (via init type)
TensorInitPtr getTensorInit(TensorInitType type, mlir::Type elmType,
                            int seed = 0);

// Return an initializer smart pointer (via string init)
TensorInitPtr getTensorInit(llvm::StringRef type, mlir::Type elmType,
                            int seed = 0);

#endif // TPP_TENSORINIT_H
