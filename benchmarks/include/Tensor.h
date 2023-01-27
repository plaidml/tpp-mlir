#pragma once

#include "llvm/ADT/STLExtras.h"

#include <cassert>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <vector>

namespace {
using ListArg = std::vector<unsigned>;
using InitArg = std::initializer_list<unsigned>;
} // namespace

/// Tensor: represents a tensor in memory, basically a pointer to data plus
/// rank and dimension.
template <class T> class Tensor {
protected:
  /// Dimensions
  ListArg dims;
  /// Size (dim0 x dim1 x ...)
  size_t size;
  /// Pointer to data
  T *data;
  /// Data size in bytes
  size_t dataSize;

  void alloc() {
    dataSize = size * sizeof(T);
    data = (T *)std::malloc(dataSize);
  }

public:
  /// Empty constructor, from dims vector
  Tensor(ListArg &dims) : dims(dims) {
    size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
    alloc();
  }

  /// Empty constructor, from dims list
  Tensor(InitArg dims) : dims(dims) {
    size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
    alloc();
  }

  /// Move constructor
  Tensor(Tensor<T> &&other)
      : dims(std::move(other.dims)), size(std::exchange(other.size, 0)),
        data(std::exchange(other.data, nullptr)),
        dataSize(std::exchange(other.dataSize, 0)) {}

  /// Copy constructor
  Tensor(const Tensor<T> &other) : dims(other.dims), size(other.size) {

    alloc();
    std::memcpy(data, other.data, dataSize);
  }

  /// Assignment operator
  Tensor<T> &operator=(const Tensor<T> &other) {
    dims = other.dims;
    size = other.size;
    /// Release current data before copying a new buffer
    if (data)
      free(data);
    alloc();
    std::memcpy(data, other.data, dataSize);
    return *this;
  }

  /// Destructor
  ~Tensor<T>() {
    // Data may have been moved elsewhere
    if (data)
      free(data);
  }

  /// Get RO data
  const T *getData() {
    // unique_ptr -> vector.data
    return data->data;
  }

  /// RW access to a single element
  T &operator[](size_t index) { return data[index]; }

  /// Get dimensions
  const ListArg &getDims() { return dims; }

  /// Get specific dimension
  unsigned getDim(unsigned i) const { return dims[i]; }

  /// Get rank
  unsigned getRank() const { return dims.size(); }

  /// Zeroes the tensor
  void clear() { memset(&data[0], 0, size * sizeof(T)); }

  /// Compares two tensors with an epsilon
  bool compareAlmostEq(Tensor &other, T epsilon) const {
    if (size != other.size || dims == other.dims)
      return false;
    // Compare delta against epsilon
    for (size_t i = 0; i < size; i++) {
      if (std::abs(data[i] - other.data[i]) > epsilon)
        return false;
    }
    return true;
  }

  /// Compares two tensors
  bool operator==(const Tensor &other) const {
    return size == other.size && dims == other.dims &&
           memcmp(data, other.data, dataSize) == 0;
  }

  /// Compares two tensors
  bool operator!=(const Tensor &other) const { return !(*this == other); }

  /// Output the tensor
  friend std::ostream &operator<<(std::ostream &out, const Tensor<T> &t) {
    out << "tensor<";
    bool first = true;
    for (auto dim : t.dims) {
      if (!first)
        out << "x";
      out << dim;
      first = false;
    }
    out << "> = [";
    for (size_t i = 0; i < t.size; i++) {
      if (i)
        out << " ";
      out << t.data[i];
    }
    out << "]";
    return out;
  }
};

/// Empty tensor of type T, initialized with zeroes.
template <class T> struct EmptyTensor : public Tensor<T> {
  EmptyTensor(InitArg dims) : Tensor<T>(dims) { this->clear(); }
  std::ostream &operator<<(std::ostream &out) {
    return out << static_cast<Tensor<T>>(this);
  }
};

/// Constant tensor of type T, initialized with incremental values.
template <class T> struct ConstantTensor : public Tensor<T> {
  ConstantTensor(InitArg dims) : Tensor<T>(dims) {
    T datum = 1; // Cast to right numeric type on assignment
    for (size_t i = 0; i < this->size; i++)
      this->data[i] = datum++;
  }
  std::ostream &operator<<(std::ostream &out) {
    return out << static_cast<Tensor<T>>(this);
  }
};

// TODO: Create random init
