// RUN: tpp-opt %s -simplify-pack -split-input-file | FileCheck %s

// CHECK-LABEL: empty_static
func.func @empty_static() -> tensor<64x16x32x32xf32> {
  // CHECK-NOT: tensor.pack
  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<64x16x32x32xf32>
  // CHECK-NEXT: return %[[EMPTY]] : tensor<64x16x32x32xf32>
  %0 = tensor.empty() : tensor<2048x512xf32>
  %1 = tensor.empty() : tensor<64x16x32x32xf32>
  %pack = tensor.pack %0 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %1 : tensor<2048x512xf32> -> tensor<64x16x32x32xf32>
  return %pack : tensor<64x16x32x32xf32>
}

// -----

// CHECK-LABEL: empty_partially_dynamic
func.func @empty_partially_dynamic(%tile1: index, %tile2: index) -> tensor<16x16x?x?xf32> {
  // CHECK-NOT: tensor.pack
  // CHECK: %[[EMPTY:.+]] = tensor.empty(%{{.+}}, %{{.+}}) : tensor<16x16x?x?xf32>
  // CHECK-NEXT: return %[[EMPTY]] : tensor<16x16x?x?xf32>
  %0 = tensor.empty() : tensor<128x128xf32>
  %1 = tensor.empty(%tile1, %tile2) : tensor<16x16x?x?xf32>
  %pack = tensor.pack %0 inner_dims_pos = [0, 1] inner_tiles = [%tile1, %tile2] into %1 : tensor<128x128xf32> -> tensor<16x16x?x?xf32>
  return %pack : tensor<16x16x?x?xf32>
}

// -----

// CHECK-LABEL: empty_fully_dynamic
func.func @empty_fully_dynamic(%tile1: index, %tile2: index, %tile3: index, %tile4: index,
                               %i: index, %j: index) -> tensor<?x?x?x?xf32> {
  // CHECK-NOT: tensor.pack
  // CHECK: %[[EMPTY:.+]] = tensor.empty(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) : tensor<?x?x?x?xf32>
  // CHECK-NEXT: return %[[EMPTY]] : tensor<?x?x?x?xf32>
  %0 = tensor.empty(%i, %j) : tensor<?x?xf32>
  %1 = tensor.empty(%tile1, %tile2, %tile3, %tile4) : tensor<?x?x?x?xf32>
  %pack = tensor.pack %0 inner_dims_pos = [0, 1] inner_tiles = [%tile1, %tile2] into %1 : tensor<?x?xf32> -> tensor<?x?x?x?xf32>
  return %pack : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: noop_pack
// CHECK-SAME: %[[ARG0:.+]]: tensor<32x32xbf16>, %[[ARG1:.+]]: tensor<1x1x32x32xbf16>
func.func @noop_pack(%arg0: tensor<32x32xbf16>, %arg1: tensor<1x1x32x32xbf16>) -> tensor<1x1x32x32xbf16> {
  // CHECK-NOT: tensor.pack
  %0 = tensor.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %arg1 
    : tensor<32x32xbf16> -> tensor<1x1x32x32xbf16>
  // CHECK: %[[EXP:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1, 2], [3]] 
  // CHECK-SAME:  : tensor<32x32xbf16> into tensor<1x1x32x32xbf16>
  // CHECK-NEXT: return %[[EXP]] : tensor<1x1x32x32xbf16> 
  return %0 : tensor<1x1x32x32xbf16>
}

// -----

// CHECK-LABEL: noop_pack_1
// CHECK-SAME: %[[ARG0:.+]]: tensor<32x32xbf16>, %[[ARG1:.+]]: tensor<1x1x32x32xbf16>
func.func @noop_pack_1(%arg0: tensor<32x32xbf16>, %arg1: tensor<1x1x32x32xbf16>) -> tensor<1x1x32x32xbf16> {
  // CHECK-NOT: tensor.pack
  %0 = tensor.pack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %arg1
    : tensor<32x32xbf16> -> tensor<1x1x32x32xbf16>
  // CHECK: %[[EXP:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1, 2], [3]]
  // CHECK-SAME:  : tensor<32x32xbf16> into tensor<1x1x32x32xbf16>
  // CHECK-NEXT: return %[[EXP]] : tensor<1x1x32x32xbf16>
  return %0 : tensor<1x1x32x32xbf16>
}

// -----

// CHECK-LABEL: op_pack_2
func.func @op_pack_2(%arg0: tensor<30x30xbf16>, %arg1: tensor<1x1x32x32xbf16>) -> tensor<1x1x32x32xbf16> {
  %pad = arith.constant 0.0 : bf16
  // CHECK: tensor.pack
  %0 = tensor.pack %arg0 padding_value(%pad : bf16) outer_dims_perm = [1, 0] 
    inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %arg1
    : tensor<30x30xbf16> -> tensor<1x1x32x32xbf16>
  // CHECK-NOT: tensor.expand_shape
  return %0 : tensor<1x1x32x32xbf16>
}

// -----

// CHECK-LABEL: op_pack_3
func.func @op_pack_3(%arg0: tensor<32x64xbf16>, %arg1: tensor<1x2x32x32xbf16>) -> tensor<1x2x32x32xbf16> {
  // We cannot simplify the pack, dropping dimension 0 would mean the following pack:
  // %arg0 inner_dims_pos = [1] inner_tiles = [32] -> 32x2x32xbf16
  // which is different from 2x32x32xbf16
  // CHECK: tensor.pack
  %0 = tensor.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %arg1
    : tensor<32x64xbf16> -> tensor<1x2x32x32xbf16>
  // CHECK-NOT: tensor.expand_shape
  return %0 : tensor<1x2x32x32xbf16>
}

// -----

// CHECK-LABEL: op_pack_4
func.func @op_pack_4(%arg0: tensor<32x64x64xbf16>, %arg1: tensor<1x2x2x32x32x32xbf16>) -> tensor<1x2x2x32x32x32xbf16> {
  // We cannot simplify the pack, dropping dimension 0, would mean the following pack:
  // %arg0 inner_dims_pos = [1, 2] inner_tiles = [32, 32] -> 32x2x2x32x32xbf16
  // which is different from 2x2x32x32x32xbf16.
  // CHECK: tensor.pack
  %0 = tensor.pack %arg0 inner_dims_pos = [0, 1, 2] inner_tiles = [32, 32, 32] into %arg1
    : tensor<32x64x64xbf16> -> tensor<1x2x2x32x32x32xbf16>
  // CHECK-NOT: tensor.expand_shape
  return %0 : tensor<1x2x2x32x32x32xbf16>
}

// -----

// CHECK-LABEL: op_pack_5
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x32xbf16>, %[[ARG1:.+]]: tensor<1x1x32x32xbf16>
// This should reshape. What about dynamic tiles?
func.func @op_pack_5(%arg0: tensor<?x32xbf16>, %arg1: tensor<1x1x32x32xbf16>) -> tensor<1x1x32x32xbf16> {
  // CHECK: tensor.pack
  // We bail out as we have unknown dim.
  %0 = tensor.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %arg1
    : tensor<?x32xbf16> -> tensor<1x1x32x32xbf16>
  // CHECK-NOT: tensor.expand_shape
  return %0 : tensor<1x1x32x32xbf16>
}

// -----

// CHECK-LABEL: rank_reduce_pack
// CHECK-SAME: %[[ARG0:.+]]: tensor<32x32xbf16>, %[[ARG1:.+]]: tensor<1x16x32x2xbf16>
func.func @rank_reduce_pack(%arg0: tensor<32x32xbf16>, %arg1: tensor<1x16x32x2xbf16>) -> tensor<1x16x32x2xbf16> {
  // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<16x32x2xbf16>
  // CHECK: %[[PACK:.+]] = tensor.pack %[[ARG0]] inner_dims_pos = [0] inner_tiles = [2] into %[[EMPTY]] 
  // CHECK-SAME:  : tensor<32x32xbf16> -> tensor<16x32x2xbf16>
  // CHECK: %[[EXP:.+]] = tensor.expand_shape %[[PACK]] {{\[}}[0, 1], [2], [3]]
  // CHECK-SAME:  : tensor<16x32x2xbf16> into tensor<1x16x32x2xbf16>
  %expanded = tensor.expand_shape %arg0 [[0, 1], [2]] : tensor<32x32xbf16> into tensor<1x32x32xbf16>
  %pack = tensor.pack %expanded inner_dims_pos = [1] inner_tiles = [2] into %arg1 
    : tensor<1x32x32xbf16> -> tensor<1x16x32x2xbf16>
  return %pack : tensor<1x16x32x2xbf16>
}

// -----

#map = affine_map<(d0) -> (d0 * 32)>

func.func @vnni_pack(%arg0: tensor<1024x512xbf16>, %arg1: tensor<16x32x16x32x2xbf16>) -> tensor<16x32x16x32x2xbf16> {
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  %0 = tensor.empty() : tensor<16x32x32x32xbf16>
  %1 = scf.forall (%arg2, %arg3) in (%c16, %c32) shared_outs(%arg4 = %arg1) -> (tensor<16x32x16x32x2xbf16>) {
    %2 = affine.apply #map(%arg3)
    %3 = affine.apply #map(%arg2)
    %extracted_slice = tensor.extract_slice %arg0[%2, %3] [32, 32] [1, 1] : tensor<1024x512xbf16> to tensor<32x32xbf16>
    %extracted_slice_0 = tensor.extract_slice %0[%arg2, %arg3, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<16x32x32x32xbf16> to tensor<1x1x32x32xbf16>
    %pack = tensor.pack %extracted_slice outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %extracted_slice_0 : tensor<32x32xbf16> -> tensor<1x1x32x32xbf16>
    %extracted_slice_1 = tensor.extract_slice %arg4[%arg2, %arg3, %c0, %c0, 0] [1, 1, %c16, %c32, 2] [1, 1, 1, 1, 1] : tensor<16x32x16x32x2xbf16> to tensor<1x1x?x?x2xbf16>
    %pack_2 = tensor.pack %pack inner_dims_pos = [2] inner_tiles = [2] into %extracted_slice_1 : tensor<1x1x32x32xbf16> -> tensor<1x1x?x?x2xbf16>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %pack_2 into %arg4[%arg2, %arg3, %c0, %c0, 0] [1, 1, %c16, %c32, 2] [1, 1, 1, 1, 1] : tensor<1x1x?x?x2xbf16> into tensor<16x32x16x32x2xbf16>
    }
  }
  return %1 : tensor<16x32x16x32x2xbf16>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0) -> (d0 * 32)>
// CHECK-LABEL: vnni_pack
// CHECK-SAME: %[[ARG0:.+]]: tensor<1024x512xbf16>, %[[ARG1:.+]]: tensor<16x32x16x32x2xbf16>
// CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG: %[[C16:.+]] = arith.constant 16 : index
// CHECK: %{{.+}} = scf.forall (%[[ARG2:.+]], %[[ARG3:.+]]) in (%[[C16]], %[[C32]]) shared_outs(%[[ARG4:.+]] = %[[ARG1]])
// CHECK: %[[AFFINE_APPLY:.+]] = affine.apply #[[MAP]](%[[ARG3]])
// CHECK: %[[AFFINE_APPLY_1:.+]] = affine.apply #[[MAP]](%[[ARG2]])
// CHECK: %[[SLICE:.+]] = tensor.extract_slice %arg0[%[[AFFINE_APPLY]], %[[AFFINE_APPLY_1]]] [32, 32] [1, 1] 
// CHECK-SAME:  : tensor<1024x512xbf16> to tensor<32x32xbf16>
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<16x32x2xbf16>
// CHECK: %[[PACK:.+]] = tensor.pack %[[SLICE]] inner_dims_pos = [0] inner_tiles = [2] into %[[EMPTY]] 
// CHECK-SAME:  : tensor<32x32xbf16> -> tensor<16x32x2xbf16>
