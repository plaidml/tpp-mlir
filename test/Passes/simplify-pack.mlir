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
