// RUN: tpp-opt %s -generalize-tensor-pack-unpack -split-input-file | FileCheck %s

func.func @pack_matmul_operand_b(%in: tensor<512x1024xf32>) -> tensor<32x16x32x32xf32> {
  %0 = tensor.empty() : tensor<32x16x32x32xf32>
  %1 = tensor.pack %in outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %0 
    : tensor<512x1024xf32> -> tensor<32x16x32x32xf32>
  return %1 : tensor<32x16x32x32xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0) -> (d0 * 32)>
// CHECK-LABEL: pack_matmul_operand_b
// CHECK-SAME: %[[ARG0:.+]]: tensor<512x1024xf32>
// CHECK: %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK: %[[PACKED:.+]] = tensor.empty() : tensor<32x16x32x32xf32>
// CHECK: %[[LOOP:.+]] = scf.for %[[I:.+]] = %[[C0]] to %[[C32]] step %[[C1]] iter_args(%[[ARG2:.+]] = %[[PACKED]])
// CHECK-NEXT: %[[LOOP1:.+]] = scf.for %[[J:.+]] = %[[C0]] to %[[C16]] step %[[C1]] iter_args(%[[ARG4:.+]] = %[[ARG2]])
// CHECK: %[[APPLY:.+]] = affine.apply #[[MAP]](%[[J]])
// CHECK: %[[APPLY1:.+]] = affine.apply #[[MAP]](%[[I]])
// CHECK: %[[SLICE:.+]] = tensor.extract_slice 
// CHECK-SAME:  %[[ARG0]][%[[APPLY]], %[[APPLY1]]] [32, 32] [1, 1] : tensor<512x1024xf32> to tensor<32x32xf32
// CHECK: %[[BUFF:.+]] = tensor.empty() : tensor<32x32xf32>
// CHECK: %[[TRANS:.+]] = linalg.transpose ins(%[[SLICE]] : tensor<32x32xf32>) 
// CHECK-SAME:  outs(%[[BUFF]] : tensor<32x32xf32>) permutation = [0, 1]
// CHECK: %[[INSERT:.+]] = tensor.insert_slice %[[TRANS]] 
// CHECK-SAME:  into %[[ARG4]][%[[I]], %[[J]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xf32> into tensor<32x16x32x32xf32>

// -----

func.func @pack_matmul_operand_a(%in: tensor<256x512xf32>) -> tensor<8x16x32x32xf32> {
  %0 = tensor.empty() : tensor<8x16x32x32xf32>
  %1 = tensor.pack %in inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %0 
    : tensor<256x512xf32> -> tensor<8x16x32x32xf32>
  return %1 : tensor<8x16x32x32xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0) -> (d0 * 32)>
// CHECK-LABEL: pack_matmul_operand_a
// CHECK-SAME: %[[ARG0:.+]]: tensor<256x512xf32>
// CHECK: %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG: %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK: %[[PACKED:.+]] = tensor.empty() : tensor<8x16x32x32xf32>
// CHECK: %{{.+}} = scf.for %[[I:.+]] = %[[C0]] to %[[C8]] step %[[C1]] iter_args(%[[ARG2:.+]] = %[[PACKED]])
// CHECK-NEXT: %{{.+}} = scf.for %[[J:.+]] = %[[C0]] to %[[C16]] step %[[C1]] iter_args(%[[ARG4:.+]] = %[[ARG2]])
// CHECK: %[[APPLY:.+]] = affine.apply #[[MAP]](%[[I]])
// CHECK: %[[APPLY1:.+]] = affine.apply #[[MAP]](%[[J]])
// CHECK: %[[SLICE:.+]] = tensor.extract_slice 
// CHECK-SAME:  %[[ARG0]][%[[APPLY]], %[[APPLY1]]] [32, 32] [1, 1] : tensor<256x512xf32> to tensor<32x32xf32
// CHECK: %[[BUFF:.+]] = tensor.empty() : tensor<32x32xf32>
// CHECK: %[[TRANS:.+]] = linalg.transpose ins(%[[SLICE]] : tensor<32x32xf32>) 
// CHECK-SAME:  outs(%[[BUFF]] : tensor<32x32xf32>) permutation = [0, 1]
// CHECK: %[[INSERT:.+]] = tensor.insert_slice %[[TRANS]] 
// CHECK-SAME:  into %[[ARG4]][%[[I]], %[[J]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xf32> into tensor<8x16x32x32xf32>



// -----

func.func @unpack_matmul(%in: tensor<8x32x32x32xf32>) -> tensor<256x1024xf32> {
  %0 = tensor.empty() : tensor<256x1024xf32>
  %1 = tensor.unpack %in inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %0 
    : tensor<8x32x32x32xf32> -> tensor<256x1024xf32>
  return %1: tensor<256x1024xf32>
}

// CHECK-LABEL: unpack_matmul
// CHECK-SAME:  %[[ARG0:.+]]: tensor<8x32x32x32xf32>
// CHECK: %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG: %[[C1024:.+]] = arith.constant 1024 : index
// CHECK-DAG: %[[C256:.+]] = arith.constant 256 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK: %[[UNPACKED:.+]] = tensor.empty() : tensor<256x1024xf32>
// CHECK: %{{.+}} = scf.for %[[I:.+]] = %[[C0]] to %[[C256]] step %[[C1]] iter_args(%[[ARG2:.+]] = %[[UNPACKED]])
// CHECK-NEXT: %{{.+}} = scf.for %[[J:.+]] = %[[C0]] to %[[C1024]] step %[[C1]] iter_args(%[[ARG4:.+]] = %[[ARG2]])
// CHECK: %[[APPLY:.+]] = affine.apply #map1(%arg1)
// CHECK: %[[APPLY1:.+]] = affine.apply #map1(%arg3)
// CHECK: %[[SLICE:.+]] = tensor.extract_slice 
// CHECK-SAME:  %[[ARG0]][%[[APPLY]], %[[APPLY1]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<8x32x32x32xf32> to tensor<32x32xf32>
// CHECK: %[[BUFF:.+]] = tensor.empty() : tensor<32x32xf32>
// CHECK: %[[TRANS:.+]] = linalg.transpose ins(%[[SLICE]] : tensor<32x32xf32>) outs(%[[BUFF]] : tensor<32x32xf32>) permutation = [0, 1]
// CHECK: %[[INSERT:.+]] = tensor.insert_slice %{{.+}} into %[[ARG4]][%[[I]], %[[J]]] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<256x1024xf32>
