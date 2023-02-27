#include "TPP/TensorInit.h"

using namespace mlir;

DenseElementsAttr TensorInit::get(ShapedType shape) {
  buffer.clear();
  for (size_t dim=0, rank = shape.getRank(); dim<rank; dim++)
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
  if (type == DataType::BF16)
    toBF16(buffer[index]);
}

void TensorInit::push_back(float value) {
  buffer.push_back(llvm::APFloat(value));
  if (type == DataType::BF16)
    toBF16(buffer.back());
}

float TensorInit::at(size_t index) {
  return buffer[index].convertToFloat();
}

DenseElementsAttr ConstantTensorInit::get(ShapedType shape) {
  auto floatValue = APFloat(1.0F);
  if (shape.getElementType().isBF16()) {
    bool ignored;
    floatValue.convert(APFloat::BFloat(), APFloat::rmNearestTiesToEven,
                       &ignored);
  } else {
    assert(shape.getElementType().isF32() && "Element type not supported");
  }

  // For some reason, memref global op needs dense tensor type
  // See: lib/Dialect/MemRef/IR/MemRefOps.cpp :: GlobalOp::verify
  auto tensorType =
      RankedTensorType::get(shape.getShape(), shape.getElementType());
  return mlir::DenseElementsAttr::get(tensorType, floatValue);
}

void ConstantTensorInit::fillData() {
  assert(false && "Should not be called");
}

void SimpleTensorInit::fillData() {
  assert(buffer.size() == 0 && "Buffer not empty");
  float data[3] = { 0.3f, 0.6f, 0.9f };
  for (size_t i=0; i<size; i++)
    push_back(data[i % 3]);
}

void ContinuousTensorInit::fillData() {
  assert(buffer.size() == 0 && "Buffer not empty");
  float normFactor = static_cast<float>(buffer.size());
  for (size_t i=0; i<size; i++)
    push_back(static_cast<float>(i) / normFactor);
}

void RandomTensorInit::fillData() {
  assert(buffer.size() == 0 && "Buffer not empty");
  for (size_t i=0; i<size; i++)
    push_back(next());
}

void NormalTensorInit::fillData() {
  assert(buffer.size() == 0 && "Buffer not empty");
  for (size_t i=0; i<size; i++)
    push_back(next());
}
