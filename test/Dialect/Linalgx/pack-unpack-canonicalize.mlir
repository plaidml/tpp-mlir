// RUN: tpp-opt %s -canonicalize -split-input-file | FileCheck %s

// CHECK: func.func @packUnpack(
// CHECK-SAME: %[[ARG0:.+]]: tensor<1x64x58x58xf32>)
func.func @packUnpack(%arg0: tensor<1x64x58x58xf32>) -> tensor<1x64x58x58xf32> {
  %alloc = tensor.empty() : tensor<1x2x58x58x32xf32>
  %0 = linalgx.pack %arg0 inner_dims_pos = [1] inner_tiles = [32] into %alloc : (tensor<1x64x58x58xf32> tensor<1x2x58x58x32xf32>) -> tensor<1x2x58x58x32xf32>
  %alloc1 = tensor.empty() : tensor<1x64x58x58xf32>
  %1 = linalgx.unpack %0 inner_dims_pos = [1] inner_tiles = [32] into %alloc1 : (tensor<1x2x58x58x32xf32> tensor<1x64x58x58xf32>) -> tensor<1x64x58x58xf32>
  // CHECK: %[[ARG0]] : tensor<1x64x58x58xf32>
  return %1 : tensor<1x64x58x58xf32>
}

// -----

// CHECK: func.func @packUnPack(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<1x2x58x58x32xf32>) -> tensor<1x2x58x58x32xf32> {
func.func @packUnPack(%arg0: tensor<1x2x58x58x32xf32>) -> tensor<1x2x58x58x32xf32> {
  %alloc = tensor.empty() : tensor<1x64x58x58xf32>
  %0 = linalgx.unpack %arg0 inner_dims_pos = [1] inner_tiles = [32] into %alloc : (tensor<1x2x58x58x32xf32> tensor<1x64x58x58xf32>) -> tensor<1x64x58x58xf32>
  %alloc1 = tensor.empty() : tensor<1x2x58x58x32xf32>
  %1 = linalgx.pack %0 inner_dims_pos = [1] inner_tiles = [32] into %alloc1 : (tensor<1x64x58x58xf32> tensor<1x2x58x58x32xf32>) -> tensor<1x2x58x58x32xf32>
  // CHECK: return %[[ARG0]] : tensor<1x2x58x58x32xf32>
  return %1 : tensor<1x2x58x58x32xf32>
}

// -----

// Different type expects canonicalization to fail.
// CHECK: func.func @packUnPack(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<1x2x58x58x32xf32>) -> tensor<1x2x58x58x32xf32> {
func.func @packUnPack(%arg0: tensor<1x2x58x58x32xf32>) -> tensor<1x2x58x58x32xf32> {
  %alloc = tensor.empty() : tensor<1x64x58x58xf32>
  %0 = linalgx.unpack %arg0 inner_dims_pos = [1] inner_tiles = [32] into %alloc : (tensor<1x2x58x58x32xf32> tensor<1x64x58x58xf32>) -> tensor<1x64x58x58xf32>
  %alloc1 = tensor.empty() : tensor<1x4x58x58x16xf32>
  %1 = linalgx.pack %0 inner_dims_pos = [1] inner_tiles = [16] into %alloc1 : (tensor<1x64x58x58xf32> tensor<1x4x58x58x16xf32>) -> tensor<1x2x58x58x32xf32>
  // CHECK-NOT: return %[[ARG0]] : tensor<1x2x58x58x32xf32>
  return %1 : tensor<1x2x58x58x32xf32>
}

// -----

func.func @packOneDTensor(%arg0: tensor<256xf32>) -> tensor<8x32xf32> {
  %alloc = tensor.empty() : tensor<8x32xf32>
  %0 = linalgx.pack %arg0 inner_dims_pos = [0] inner_tiles = [32] into %alloc : (tensor<256xf32> tensor<8x32xf32>) -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}
// CHECK: func.func @packOneDTensor(
// CHECK-SAME: %[[ARG0:.+]]: tensor<256xf32>) -> tensor<8x32xf32> {
// CHECK: %[[EXPANDED:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1]] : tensor<256xf32> into tensor<8x32xf32>
// CHECK: return %[[EXPANDED]] : tensor<8x32xf32>

// -----

func.func @packTensorEmpty() -> tensor<1x2x58x58x32xf32> {
  %alloc = tensor.empty() : tensor<1x64x58x58xf32>
  %alloc1 = tensor.empty() : tensor<1x2x58x58x32xf32>
  %0 = linalgx.pack %alloc inner_dims_pos = [1] inner_tiles = [32] into %alloc1 : (tensor<1x64x58x58xf32> tensor<1x2x58x58x32xf32>) -> tensor<1x2x58x58x32xf32>
  return %0 : tensor<1x2x58x58x32xf32>
}

// CHECK: func.func @packTensorEmpty() -> tensor<1x2x58x58x32xf32> {
// CHECK: %[[ALLOC:.+]] = tensor.empty() : tensor<1x2x58x58x32xf32>
// CHECK: return %[[ALLOC]] : tensor<1x2x58x58x32xf32>
