//===- TensorInitInt.cpp -----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/TensorInitInt.h"

using namespace mlir;

TensorInitInt::DataType TensorInitInt::getTensorInitDataType(mlir::Type type) {
  if (type.isInteger(8))
    return DataType::I8;
  if (type.isInteger(16))
    return DataType::I16;
  if (type.isInteger(32))
    return DataType::I32;
  if (type.isInteger(64))
    return DataType::I64;
  assert(false && "Invalid tensor init data type (only I8, I16, I32, I64)");
}

unsigned TensorInitInt::getDataTypeBitWidth(TensorInitInt::DataType type) {
  switch (type) {
  case DataType::I8:
    return 8;
  case DataType::I16:
    return 16;
  case DataType::I32:
    return 32;
  case DataType::I64:
    return 64;
  }
}

bool TensorInitInt::isDataTypeSigned(TensorInitInt::DataType type) {
  switch (type) {
  case DataType::I8:
  case DataType::I16:
  case DataType::I32:
  case DataType::I64:
    return true;
  }
}

void TensorInitInt::insert(size_t index, uint64_t value) {
  this->TensorInit::insert(index, APInt(bitWidth, value, isSigned));
}

void TensorInitInt::push(uint64_t value) {
  this->TensorInit::push(APInt(bitWidth, value, isSigned));
}

void TensorInitInt::convertType(llvm::APInt &value) {
  assert(value.getBitWidth() == bitWidth && "Invalid element size");
}

DenseElementsAttr ConstantTensorInitInt::get(ShapedType shape) {
  auto value = APInt(bitWidth, 1, isSigned);
  if (!isTypeSupported(shape.getElementType()))
    assert(false && "Element type not supported");
  convertType(value);

  // For some reason, memref global op needs dense tensor type
  // See: lib/Dialect/MemRef/IR/MemRefOps.cpp :: GlobalOp::verify
  auto tensorType =
      RankedTensorType::get(shape.getShape(), shape.getElementType());
  return mlir::DenseElementsAttr::get(tensorType, value);
}

void ConstantTensorInitInt::fillData() {
  assert(false && "Should not be called");
}

void SimpleTensorInitInt::fillData() {
  assert(buffer.size() == 0 && "Buffer not empty");
  uint64_t data[3] = {0, 1, 2};
  for (size_t i = 0; i < size; i++)
    push(data[i % 3]);
}

void ContinuousTensorInitInt::fillData() {
  assert(buffer.size() == 0 && "Buffer not empty");
  float normFactor = static_cast<float>(size);
  for (size_t i = 0; i < size; i++)
    push(static_cast<uint64_t>((static_cast<float>(i) / normFactor) *
                               upperBound));
}

void RandomTensorInitInt::fillData() {
  assert(buffer.size() == 0 && "Buffer not empty");
  for (size_t i = 0; i < size; i++)
    push(next());
}

void NormalTensorInitInt::fillData() {
  assert(buffer.size() == 0 && "Buffer not empty");
  for (size_t i = 0; i < size; i++)
    push(next());
}
