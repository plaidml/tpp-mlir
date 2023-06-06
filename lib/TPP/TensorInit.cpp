//===- TensorInit.cpp --------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/TensorInit.h"
#include "TPP/TensorInitFloat.h"

using namespace mlir;

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

  if (TensorInitFloat::isTypeSupported(elmType)) {
    auto dataType = TensorInitFloat::getTensorInitDataType(elmType);
    switch (type) {
    case TensorInitType::Constant:
      return std::make_unique<ConstantTensorInitFloat>(dataType);
    case TensorInitType::Simple:
      return std::make_unique<SimpleTensorInitFloat>(dataType);
    case TensorInitType::Continuous:
      return std::make_unique<ContinuousTensorInitFloat>(dataType);
    case TensorInitType::Random:
      assert(seed && "Can't call random initializers without seed");
      return std::make_unique<RandomTensorInitFloat>(dataType, seed);
    case TensorInitType::Normal:
      assert(seed && "Can't call random initializers without seed");
      return std::make_unique<NormalTensorInitFloat>(dataType, seed);
    default:
      assert(false && "Invalid tensor initializer type");
    }
  }

  assert(false && "Unsupported tensor element type");
  return nullptr;
}

TensorInitPtr getTensorInit(StringRef type, mlir::Type elmType, int seed) {
  auto initType = parseTensorInitType(type);
  return getTensorInit(initType, elmType, seed);
}
