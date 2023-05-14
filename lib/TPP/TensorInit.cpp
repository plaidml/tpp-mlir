//===- TensorInit.cpp --------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/TensorInit.h"

using namespace mlir;

TensorInit::DataType getTensorInitDataType(mlir::Type type) {
  if (type.isBF16())
    return TensorInit::BF16;
  if (type.isF32())
    return TensorInit::FP32;
  if (type.isF64())
    return TensorInit::FP64;
  assert(false && "Invalid tensor init data type (only FP32, FP64, BF16)");
}

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
  auto dataType = getTensorInitDataType(elmType);
  // Defaults for seed or not
  if (type == TensorInitType::Auto) {
    if (seed)
      type = TensorInitType::Normal;
    else
      type = TensorInitType::Constant;
  }
  switch (type) {
  case TensorInitType::Constant:
    return std::make_unique<ConstantTensorInit>(dataType);
  case TensorInitType::Simple:
    return std::make_unique<SimpleTensorInit>(dataType);
  case TensorInitType::Continuous:
    return std::make_unique<ContinuousTensorInit>(dataType);
  case TensorInitType::Random:
    assert(seed && "Can't call random initializers without seed");
    return std::make_unique<RandomTensorInit>(dataType, seed);
  case TensorInitType::Normal:
    assert(seed && "Can't call random initializers without seed");
    return std::make_unique<NormalTensorInit>(dataType, seed);
  default:
    assert(false && "Invalid tensor initializer type");
  }
}

TensorInitPtr getTensorInit(StringRef type, mlir::Type elmType, int seed) {
  auto initType = parseTensorInitType(type);
  return getTensorInit(initType, elmType, seed);
}

DenseElementsAttr TensorInit::get(ShapedType shape) {
  buffer.clear();
  for (size_t dim = 0, rank = shape.getRank(); dim < rank; dim++)
    size *= shape.getDimSize(dim);
  fillData();
  // For some reason, memref global op needs dense tensor type
  // See: lib/Dialect/MemRef/IR/MemRefOps.cpp :: GlobalOp::verify
  auto tensorType =
      RankedTensorType::get(shape.getShape(), shape.getElementType());
  return mlir::DenseElementsAttr::get(tensorType, buffer);
}

void TensorInit::insert(size_t index, float value) {
  buffer[index] = llvm::APFloat(value);
  convertType(buffer[index]);
}

void TensorInit::push(float value) {
  buffer.push_back(llvm::APFloat(value));
  convertType(buffer.back());
}

void TensorInit::convertType(llvm::APFloat &value) {
  switch (type) {
  case DataType::FP32:
    toFP32(value);
    break;
  case DataType::FP64:
    toFP64(value);
    break;
  case DataType::BF16:
    toBF16(value);
    break;
  }
}

float TensorInit::at(size_t index) { return buffer[index].convertToFloat(); }

DenseElementsAttr ConstantTensorInit::get(ShapedType shape) {
  auto floatValue = APFloat(1.0F);
  if (!isTypeSupported(shape.getElementType()))
    assert(false && "Element type not supported");
  convertType(floatValue);

  // For some reason, memref global op needs dense tensor type
  // See: lib/Dialect/MemRef/IR/MemRefOps.cpp :: GlobalOp::verify
  auto tensorType =
      RankedTensorType::get(shape.getShape(), shape.getElementType());
  return mlir::DenseElementsAttr::get(tensorType, floatValue);
}

void ConstantTensorInit::fillData() { assert(false && "Should not be called"); }

void SimpleTensorInit::fillData() {
  assert(buffer.size() == 0 && "Buffer not empty");
  float data[3] = {0.3f, 0.6f, 0.9f};
  for (size_t i = 0; i < size; i++)
    push(data[i % 3]);
}

void ContinuousTensorInit::fillData() {
  assert(buffer.size() == 0 && "Buffer not empty");
  float normFactor = static_cast<float>(size);
  for (size_t i = 0; i < size; i++)
    push(static_cast<float>(i) / normFactor);
}

void RandomTensorInit::fillData() {
  assert(buffer.size() == 0 && "Buffer not empty");
  for (size_t i = 0; i < size; i++)
    push(next());
}

void NormalTensorInit::fillData() {
  assert(buffer.size() == 0 && "Buffer not empty");
  for (size_t i = 0; i < size; i++)
    push(next());
}
