//===- TensorInit.cpp --------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/TensorInit.h"
#include "TPP/TensorInitFloat.h"
#include "TPP/TensorInitInt.h"

#include <functional>
#include <unordered_map>

using namespace mlir;

namespace {

struct InitKey {
  InitKey() = default;
  explicit InitKey(TensorInitType type, mlir::Type elmType, int seed)
      : type(type), elmType(elmType) {
    // Seed only matters for randomized types.
    switch (type) {
    case TensorInitType::Random:
    case TensorInitType::Normal:
      this->seed = seed;
      break;
    default:
      this->seed = 0;
      break;
    }
  }

  bool operator==(const InitKey &ik) const {
    return type == ik.type && elmType == ik.elmType && seed == ik.seed;
  }

  TensorInitType type;
  mlir::Type elmType;
  int seed;
};

struct InitKeyHash_fn {
  std::size_t operator()(const InitKey &ik) const {
    auto h1 = std::hash<TensorInitType>{}(ik.type);
    auto h2 = std::hash<mlir::Type>{}(ik.elmType);
    auto h3 = std::hash<int>{}(ik.seed);
    return h1 ^ h2 ^ h3;
  }
};

std::unordered_map<InitKey, TensorInitPtr, InitKeyHash_fn> tensorInitializers;
} // namespace

TensorInitType parseTensorInitType(StringRef name) {
  auto type = StringSwitch<TensorInitType>(name)
                  .Case("", TensorInitType::Auto)
                  .Case("const", TensorInitType::Constant)
                  .Case("simple", TensorInitType::Simple)
                  .Case("cont", TensorInitType::Continuous)
                  .Case("random", TensorInitType::Random)
                  .Case("normal", TensorInitType::Normal)
                  .Default(TensorInitType::Invalid);
  return type;
}

TensorInitPtr getTensorInit(TensorInitType type, mlir::Type elmType, int seed) {
  // Defaults for seed or not
  if (type == TensorInitType::Auto) {
    if (seed)
      type = TensorInitType::Normal;
    else
      type = TensorInitType::Constant;
  }

  InitKey key(type, elmType, seed);
  if (tensorInitializers.find(key) != tensorInitializers.end())
    return tensorInitializers[key];

  TensorInitPtr initPtr = nullptr;

  if (TensorInitFloat::isTypeSupported(elmType)) {
    auto dataType = TensorInitFloat::getTensorInitDataType(elmType);
    switch (type) {
    case TensorInitType::Constant:
      initPtr = std::make_unique<ConstantTensorInitFloat>(dataType);
    case TensorInitType::Simple:
      initPtr = std::make_unique<SimpleTensorInitFloat>(dataType);
    case TensorInitType::Continuous:
      initPtr = std::make_unique<ContinuousTensorInitFloat>(dataType);
    case TensorInitType::Random:
      assert(seed && "Can't call random initializers without seed");
      initPtr = std::make_unique<RandomTensorInitFloat>(dataType, seed);
    case TensorInitType::Normal:
      assert(seed && "Can't call random initializers without seed");
      initPtr = std::make_unique<NormalTensorInitFloat>(dataType, seed);
    default:
      assert(false && "Invalid tensor initializer type");
    }
  }

  if (TensorInitInt::isTypeSupported(elmType)) {
    auto dataType = TensorInitInt::getTensorInitDataType(elmType);
    switch (type) {
    case TensorInitType::Constant:
      initPtr = std::make_unique<ConstantTensorInitInt>(dataType);
    case TensorInitType::Simple:
      initPtr = std::make_unique<SimpleTensorInitInt>(dataType);
    case TensorInitType::Continuous:
      initPtr = std::make_unique<ContinuousTensorInitInt>(dataType);
    case TensorInitType::Random:
      assert(seed && "Can't call random initializers without seed");
      initPtr = std::make_unique<RandomTensorInitInt>(dataType, seed);
    case TensorInitType::Normal:
      assert(seed && "Can't call random initializers without seed");
      initPtr = std::make_unique<NormalTensorInitInt>(dataType, seed);
    default:
      assert(false && "Invalid tensor initializer type");
    }
  }

  assert(initPtr && "Unsupported tensor element type");
  tensorInitializers[key] = initPtr;

  return initPtr;
}

TensorInitPtr getTensorInit(StringRef type, mlir::Type elmType, int seed) {
  auto initType = parseTensorInitType(type);
  return getTensorInit(initType, elmType, seed);
}
