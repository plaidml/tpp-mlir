// RUN: tpp-opt %s -generalize-tensor-pack-unpack -split-input-file | FileCheck %s

func.func @pack_matmul_operand_b(%in: tensor<512x1024xf32>) -> tensor<32x16x32x32xf32> {
  %0 = tensor.empty() : tensor<32x16x32x32xf32>
  %1 = tensor.pack %in outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %0 
    : tensor<512x1024xf32> -> tensor<32x16x32x32xf32>
  return %1 : tensor<32x16x32x32xf32>
}

// CHECK-LABEL: pack_matmul_operand_b
// CHECK-SAME: %[[ARG0:.+]]: tensor<512x1024xf32>
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<32x16x32x32xf32>
// CHECK: %[[EXP:.+]] = tensor.expand_shape %arg0 {{\[}}[0, 1], [2, 3]] : tensor<512x1024xf32> 
// CHECK-SAME:  into tensor<16x32x32x32xf32>
// CHECK: %[[T:.+]] = linalg.transpose ins(%expanded : tensor<16x32x32x32xf32>) outs(%[[EMPTY]] : tensor<32x16x32x32xf32>) 
// CHECK-SAME:  permutation = [2, 0, 1, 3]

// -----

func.func @pack_matmul_operand_a(%in: tensor<256x512xf32>) -> tensor<8x16x32x32xf32> {
  %0 = tensor.empty() : tensor<8x16x32x32xf32>
  %1 = tensor.pack %in inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %0 
    : tensor<256x512xf32> -> tensor<8x16x32x32xf32>
  return %1 : tensor<8x16x32x32xf32>
}

// CHECK-LABEL: pack_matmul_operand_a
// CHECK-SAME: %[[ARG0:.+]]: tensor<256x512xf32>
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<8x16x32x32xf32>
// CHECK: %[[EXP:.+]] = tensor.expand_shape %arg0 {{\[}}[0, 1], [2, 3]] : tensor<256x512xf32> 
// CHECK-SAME:  into tensor<8x32x16x32xf32> 
// CHECK: %[[T:.+]] = linalg.transpose ins(%[[EXP]] : tensor<8x32x16x32xf32>) outs(%[[EMPTY]] : tensor<8x16x32x32xf32>) 
// CHECK-SAME:  permutation = [0, 2, 1, 3]

// -----

func.func @unpack_matmul(%in: tensor<8x32x32x32xf32>) -> tensor<256x1024xf32> {
  %0 = tensor.empty() : tensor<256x1024xf32>
  %1 = tensor.unpack %in inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %0 
    : tensor<8x32x32x32xf32> -> tensor<256x1024xf32>
  return %1: tensor<256x1024xf32>
}

// CHECK-LABEL: unpack_matmul
// CHECK-SAME: %[[ARG0:.+]]: tensor<8x32x32x32xf32>
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<8x32x32x32xf32>
// CHECK: %[[T:.+]] = linalg.transpose ins(%[[ARG0]] : tensor<8x32x32x32xf32>) outs(%[[EMPTY]] : tensor<8x32x32x32xf32>) 
// CHECK-SAME:  permutation = [0, 2, 1, 3]
// CHECK: %[[CLP:.+]] = tensor.collapse_shape %[[T]] {{\[}}[0, 1], [2, 3]] : tensor<8x32x32x32xf32> into tensor<256x1024xf32>
