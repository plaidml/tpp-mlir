// RUN: tpp-opt -pack-vnni %s | FileCheck %s

func.func @matmul(%arg0: tensor<32x4x4xbf16>, %arg1: tensor<32x4x4xbf16>, %arg2: tensor<4x4xbf16>) -> tensor<4x4xbf16>{
  %0 = linalg.batch_reduce_matmul ins(%arg0, %arg1:tensor<32x4x4xbf16>, tensor<32x4x4xbf16>) 
                                  outs(%arg2:tensor<4x4xbf16>) -> tensor<4x4xbf16>
  return %0: tensor<4x4xbf16>
}

// CHECK-LABEL: matmul
// CHECK-SAME: %[[ARG0:.+]]: tensor<32x4x4xbf16>, %[[ARG1:.+]]: tensor<32x4x4xbf16>, %[[ARG2:.+]]: tensor<4x4xbf16>
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<32x2x4x2xbf16>
// CHECK: %[[PACK:.+]] = tensor.pack %[[ARG1]] 
// CHECK-SAME:  inner_dims_pos = [1] inner_tiles = [2] into %[[EMPTY]] : tensor<32x4x4xbf16> -> tensor<32x2x4x2xbf16>
// CHECK: tpp.brgemm (%[[ARG0]] : tensor<32x4x4xbf16>, %[[PACK]] : tensor<32x2x4x2xbf16>, %[[ARG2]] : tensor<4x4xbf16>) -> (tensor<4x4xbf16>)
