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

func.func @unpack_matmul(%in: tensor<8x32x32x32xf32>, %dest: tensor<256x1024xf32>) -> tensor<256x1024xf32> {
  %1 = tensor.unpack %in inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %dest 
    : tensor<8x32x32x32xf32> -> tensor<256x1024xf32>
  return %1: tensor<256x1024xf32>
}

// CHECK-LABEL: unpack_matmul
// CHECK-SAME: %[[ARG0:.+]]: tensor<8x32x32x32xf32>
// CHECK-SAME: %[[ARG1:.+]]: tensor<256x1024xf32>
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<8x32x32x32xf32>
// CHECK: %[[T:.+]] = linalg.transpose ins(%[[ARG0]] : tensor<8x32x32x32xf32>) outs(%[[EMPTY]] : tensor<8x32x32x32xf32>) 
// CHECK-SAME:  permutation = [0, 2, 1, 3]
// CHECK: %[[CLP:.+]] = tensor.collapse_shape %[[T]] {{\[}}[0, 1], [2, 3]] : tensor<8x32x32x32xf32> into tensor<256x1024xf32>
// CHECK: %{{.+}} = linalg.copy ins(%[[CLP]] : tensor<256x1024xf32>) outs(%[[ARG1]] : tensor<256x1024xf32>)

// -----

func.func @unpack_conv(%in: tensor<1x4x6x6x2xf32>, %out: tensor<1x6x6x8xf32>) -> tensor<1x6x6x8xf32> {
  %1 = tensor.unpack %in outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [2] into %out : tensor<1x4x6x6x2xf32> -> tensor<1x6x6x8xf32>
  return %1: tensor<1x6x6x8xf32>
}

// CHECK-LABEL: unpack_conv
// CHECK-SAME: %[[IN:.+]]: tensor<1x4x6x6x2xf32>, %[[OUT:.+]]: tensor<1x6x6x8xf32>
// CHECK-DAG: %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG: %[[C6:.+]] = arith.constant 6 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK: %[[OUTER:.+]] = scf.for %[[ARG2:.+]] = %[[C0]] to %[[C6]] step %[[C1]] iter_args(%[[ARG3:.+]] = %[[OUT]])
// CHECK-NEXT: %[[MIDDLE:.+]] = scf.for %[[ARG4:.+]] = %[[C0]] to %[[C6]] step %[[C1]] iter_args(%[[ARG5:.+]] = %[[ARG3]])
// CHECK-NEXT: %[[INNER:.+]] = scf.for %[[ARG6:.+]] = %[[C0]] to %[[C8]] step %[[C1]] iter_args(%[[ARG7:.+]] = %[[ARG5]])
// CHECK: %[[APPLY:.+]] = affine.apply #map(%[[ARG6]])
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<1x1x1x2xf32>
// CHECK: %[[APPLY_1:.+]] = affine.apply #map1(%[[ARG6]])
// CHECK: %[[SLICE:.+]] = tensor.extract_slice %[[IN]][0, %[[APPLY_1]], %[[ARG2]], %[[ARG4]], 0] [1, 1, 1, 1, 2] [1, 1, 1, 1, 1] : tensor<1x4x6x6x2xf32> to tensor<2xf32>
// CHECK: %[[EMPTY1:.+]] = tensor.empty() : tensor<2xf32>
// CHECK: %[[T:.+]] = linalg.transpose ins(%[[SLICE]] : tensor<2xf32>) outs(%[[EMPTY1]] : tensor<2xf32>) permutation = [0]
// CHECK: %[[INSERT:.+]] = tensor.insert_slice %[[T]] into %[[EMPTY]][0, 0, 0, 0] [1, 1, 1, 2] [1, 1, 1, 1] : tensor<2xf32> into tensor<1x1x1x2xf32>
// CHECK: %[[SLICE_1:.+]] = tensor.extract_slice %[[INSERT]][0, 0, 0, %[[APPLY]]] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x2xf32> to tensor<1x1x1x1xf32>
// CHECK: %[[SLICE_DEST:.+]] tensor.insert_slice %{{.+}} into %[[ARG7]][0, %[[ARG2]], %[[ARG4]], %[[ARG6]]] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<1x6x6x8xf32>
