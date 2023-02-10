// RUN: tpp-opt %s -propagate-pack-and-unpack -canonicalize -split-input-file | FileCheck -check-prefix=CHECK %s

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>

func.func @matmul(%arg0: tensor<128x512xf32>, %arg1: tensor<512x256xf32>, %arg2: tensor<128x256xf32>) -> tensor<128x256xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<4x16x32x32xf32>
  %pack = tensor.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %0 : tensor<128x512xf32> -> tensor<4x16x32x32xf32>
  %1 = tensor.empty() : tensor<8x16x32x32xf32>
  %pack_0 = tensor.pack %arg1 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %1 : tensor<512x256xf32> -> tensor<8x16x32x32xf32>
  %2 = tensor.empty() : tensor<4x8x32x32xf32>
  %pack_1 = tensor.pack %arg2 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %2 : tensor<128x256xf32> -> tensor<4x8x32x32xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack, %pack_0 : tensor<4x16x32x32xf32>, tensor<8x16x32x32xf32>) outs(%pack_1 : tensor<4x8x32x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %5 = arith.mulf %in, %in_2 : f32
      %6 = arith.addf %out, %5 : f32
      linalg.yield %6 : f32
  } -> tensor<4x8x32x32xf32>
  %unpack = tensor.unpack %3 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %arg2 : tensor<4x8x32x32xf32> -> tensor<128x256xf32>
  %4 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%unpack : tensor<128x256xf32>) {
    ^bb0(%out: f32):
      %5 = arith.maxf %out, %cst : f32
      linalg.yield %5 : f32
  } -> tensor<128x256xf32>
  return %4 : tensor<128x256xf32>
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: func.func @matmul(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<128x512xf32>, 
// CHECK-SAME:  %[[ARG1:.+]]: tensor<512x256xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: tensor<128x256xf32>) -> tensor<128x256xf32> {
// CHECK: %[[BUFF0:.+]] = tensor.empty() : tensor<4x16x32x32xf32>
// CHECK: %[[PACK0:.+]] = tensor.pack %[[ARG0]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUFF0]] : tensor<128x512xf32> -> tensor<4x16x32x32xf32>
// CHECK: %[[BUFF1:.+]] = tensor.empty() : tensor<8x16x32x32xf32>
// CHECK: %[[PACK1:.+]] = tensor.pack %[[ARG1]] outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUFF1]] : tensor<512x256xf32> -> tensor<8x16x32x32xf32>
// CHECK: %[[BUFF2:.+]] = tensor.empty() : tensor<4x8x32x32xf32>
// CHECK: %[[PACK2:.+]] = tensor.pack %[[ARG2]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUFF2]] : tensor<128x256xf32> -> tensor<4x8x32x32xf32>
// CHECK: %[[VAL:.+]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%[[PACK0]], %[[PACK1]] : tensor<4x16x32x32xf32>, tensor<8x16x32x32xf32>) outs(%[[PACK2]] : tensor<4x8x32x32xf32>)
// CHECK: %[[VAL1:.+]] = linalg.generic {indexing_maps = [#[[MAP3]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%[[VAL]] : tensor<4x8x32x32xf32>)
// CHECK: %[[OUT:.+]] = tensor.unpack %[[VAL1]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[ARG2]] : tensor<4x8x32x32xf32> -> tensor<128x256xf32>
// CHECK: return %[[OUT]] : tensor<128x256xf32>
// CHECK: }

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>

func.func @matmul(%arg0: tensor<128x512xf32>, %arg1: tensor<512x256xf32>, %arg2: tensor<128x256xf32>) -> tensor<128x256xf32> {
  %0 = tensor.empty() : tensor<4x16x32x32xf32>
  %pack = tensor.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %0 : tensor<128x512xf32> -> tensor<4x16x32x32xf32>
  %1 = tensor.empty() : tensor<8x16x32x32xf32>
  %pack_0 = tensor.pack %arg1 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %1 : tensor<512x256xf32> -> tensor<8x16x32x32xf32>
  %2 = tensor.empty() : tensor<4x8x32x32xf32>
  %pack_1 = tensor.pack %arg2 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %2 : tensor<128x256xf32> -> tensor<4x8x32x32xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack, %pack_0 : tensor<4x16x32x32xf32>, tensor<8x16x32x32xf32>) outs(%pack_1 : tensor<4x8x32x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %5 = arith.mulf %in, %in_2 : f32
      %6 = arith.addf %out, %5 : f32
      linalg.yield %6 : f32
  } -> tensor<4x8x32x32xf32>
  %unpack = tensor.unpack %3 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %arg2 : tensor<4x8x32x32xf32> -> tensor<128x256xf32>
  %4 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%unpack : tensor<128x256xf32>) outs(%arg2 : tensor<128x256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %5 = arith.addf %in, %out : f32
      linalg.yield %5 : f32
  } -> tensor<128x256xf32>
  return %4 : tensor<128x256xf32>
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: func.func @matmul(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<128x512xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: tensor<512x256xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: tensor<128x256xf32>) -> tensor<128x256xf32> {
// CHECK: %[[BUFF0:.+]] = tensor.empty() : tensor<4x16x32x32xf32>
// CHECK: %[[PACK0:.+]] = tensor.pack %[[ARG0]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUFF0]] : tensor<128x512xf32> -> tensor<4x16x32x32xf32>
// CHECK: %[[BUFF1:.+]] = tensor.empty() : tensor<8x16x32x32xf32>
// CHECK: %[[PACK1:.+]] = tensor.pack %[[ARG1]] outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUFF1]] : tensor<512x256xf32> -> tensor<8x16x32x32xf32>
// CHECK: %[[BUFF2:.+]] = tensor.empty() : tensor<4x8x32x32xf32>
// CHECK: %[[PACK2:.+]] = tensor.pack %[[ARG2]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUFF2]] : tensor<128x256xf32> -> tensor<4x8x32x32xf32>
// CHECK: %[[VAL:.+]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%[[PACK0]], %[[PACK1]] : tensor<4x16x32x32xf32>, tensor<8x16x32x32xf32>) outs(%[[PACK2]] : tensor<4x8x32x32xf32>)
// CHECK: %[[BUFF2_2:.+]] = tensor.empty() : tensor<4x8x32x32xf32>
// CHECK: %[[PACK2_2:.+]] = tensor.pack %[[ARG2]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUFF2_2]] : tensor<128x256xf32> -> tensor<4x8x32x32xf32>
// CHECK: %[[VAL1:.+]] = linalg.generic {indexing_maps = [#[[MAP3]], #[[MAP3]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[VAL]] : tensor<4x8x32x32xf32>) outs(%[[PACK2_2]] : tensor<4x8x32x32xf32>)
// CHECK: %[[OUT:.+]] = tensor.unpack %[[VAL1]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[ARG2]] : tensor<4x8x32x32xf32> -> tensor<128x256xf32>
// CHECK: return %[[OUT]] : tensor<128x256xf32>
// CHECK: }

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @conv(%arg0: tensor<1x56x56x64xf32>, %arg1: tensor<1x1x64x64xf32>, %arg2: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x2x56x56x32xf32>
  %pack = tensor.pack %arg0 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %0 : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
  %1 = tensor.empty() : tensor<2x2x1x1x32x32xf32>
  %pack_0 = tensor.pack %arg1 outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %1 : tensor<1x1x64x64xf32> -> tensor<2x2x1x1x32x32xf32>
  %2 = tensor.empty() : tensor<1x2x56x56x32xf32>
  %pack_1 = tensor.pack %arg2 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %2 : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%pack, %pack_0 : tensor<1x2x56x56x32xf32>, tensor<2x2x1x1x32x32xf32>) outs(%pack_1 : tensor<1x2x56x56x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %5 = arith.mulf %in, %in_2 : f32
      %6 = arith.addf %out, %5 : f32
      linalg.yield %6 : f32
  } -> tensor<1x2x56x56x32xf32>
  %unpack = tensor.unpack %3 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %arg2 : tensor<1x2x56x56x32xf32> -> tensor<1x56x56x64xf32>
    %4 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%unpack : tensor<1x56x56x64xf32>) {
    ^bb0(%out: f32):
      %5 = arith.maxf %out, %cst : f32
      linalg.yield %5 : f32
  } -> tensor<1x56x56x64xf32>
  return %4 : tensor<1x56x56x64xf32>
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK: func.func @conv(
// CHECK-SAME: %[[ARG0:.+]]: tensor<1x56x56x64xf32>,
// CHECK-SAME: %[[ARG1:.+]]: tensor<1x1x64x64xf32>,
// CHECK-SAME: %[[ARG2:.+]]: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32> {
// CHECK: %[[BUFF0:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK0:.+]] = tensor.pack %[[ARG0]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF0]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CHECK: %[[BUFF1:.+]] = tensor.empty() : tensor<2x2x1x1x32x32xf32>
// CHECK: %[[PACK1:.+]] = tensor.pack %[[ARG1]] outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %[[BUFF1]] : tensor<1x1x64x64xf32> -> tensor<2x2x1x1x32x32xf32>
// CHECK: %[[BUFF2:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK2:.+]] = tensor.pack %[[ARG2]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF2]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CHECK: %[[VAL:.+]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%[[PACK0]], %[[PACK1]] : tensor<1x2x56x56x32xf32>, tensor<2x2x1x1x32x32xf32>) outs(%[[PACK2]] : tensor<1x2x56x56x32xf32>)
// CHECK: %[[VAL1:.+]] = linalg.generic {indexing_maps = [#[[MAP3]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} outs(%[[VAL]] : tensor<1x2x56x56x32xf32>)
// CHECK: %[[UNPACK:.+]] = tensor.unpack %[[VAL1]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[ARG2]] : tensor<1x2x56x56x32xf32> -> tensor<1x56x56x64xf32>
// CHECK: return %[[UNPACK]] : tensor<1x56x56x64xf32>

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @conv(%arg0: tensor<1x56x56x64xf32>, %arg1: tensor<1x1x64x64xf32>, %arg2: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32> {
  %0 = tensor.empty() : tensor<1x2x56x56x32xf32>
  %pack = tensor.pack %arg0 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %0 : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
  %1 = tensor.empty() : tensor<2x2x1x1x32x32xf32>
  %pack_0 = tensor.pack %arg1 outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %1 : tensor<1x1x64x64xf32> -> tensor<2x2x1x1x32x32xf32>
  %2 = tensor.empty() : tensor<1x2x56x56x32xf32>
  %pack_1 = tensor.pack %arg2 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %2 : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%pack, %pack_0 : tensor<1x2x56x56x32xf32>, tensor<2x2x1x1x32x32xf32>) outs(%pack_1 : tensor<1x2x56x56x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %5 = arith.mulf %in, %in_2 : f32
      %6 = arith.addf %out, %5 : f32
      linalg.yield %6 : f32
  } -> tensor<1x2x56x56x32xf32>
  %unpack = tensor.unpack %3 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %arg2 : tensor<1x2x56x56x32xf32> -> tensor<1x56x56x64xf32>
    %4 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%unpack : tensor<1x56x56x64xf32>) outs(%arg2 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %5 = arith.addf %in, %out : f32
      linalg.yield %5 : f32
  } -> tensor<1x56x56x64xf32>
  return %4 : tensor<1x56x56x64xf32>
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK: func.func @conv(
// CHECK-SAME: %[[ARG0:.+]]: tensor<1x56x56x64xf32>,
// CHECK-SAME: %[[ARG1:.+]]: tensor<1x1x64x64xf32>,
// CHECK-SAME: %[[ARG2:.+]]: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32> {
// CHECK: %[[BUFF0:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK0:.+]] = tensor.pack %[[ARG0]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF0]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CHECK: %[[BUFF1:.+]] = tensor.empty() : tensor<2x2x1x1x32x32xf32>
// CHECK: %[[PACK1:.+]] = tensor.pack %[[ARG1]] outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %[[BUFF1]] : tensor<1x1x64x64xf32> -> tensor<2x2x1x1x32x32xf32>
// CHECK: %[[BUFF2:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK2:.+]] = tensor.pack %[[ARG2]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF2]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CHECK: %[[VAL:.+]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%[[PACK0]], %[[PACK1]] : tensor<1x2x56x56x32xf32>, tensor<2x2x1x1x32x32xf32>) outs(%[[PACK2]] : tensor<1x2x56x56x32xf32>)
// CHECK: %[[BUFF2_2:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK2_2:.+]] = tensor.pack %[[ARG2]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF2_2]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CHECK: %[[VAL1:.+]] = linalg.generic {indexing_maps = [#[[MAP3]], #[[MAP3]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%[[VAL]] : tensor<1x2x56x56x32xf32>) outs(%[[PACK2_2]] : tensor<1x2x56x56x32xf32>)
// CHECK: %[[UNPACK:.+]] = tensor.unpack %[[VAL1]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[ARG2]] : tensor<1x2x56x56x32xf32> -> tensor<1x56x56x64xf32>
// CHECK: return %[[UNPACK]] : tensor<1x56x56x64xf32>

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d3)>

func.func @conv(%arg0: tensor<1x56x56x64xf32>, %arg1: tensor<1x1x64x64xf32>, %arg2: tensor<1x56x56x64xf32>, %arg3: tensor<64xf32>) -> tensor<1x56x56x64xf32> {
  %0 = tensor.empty() : tensor<1x2x56x56x32xf32>
  %pack = tensor.pack %arg0 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %0 : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
  %1 = tensor.empty() : tensor<2x2x1x1x32x32xf32>
  %pack_0 = tensor.pack %arg1 outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %1 : tensor<1x1x64x64xf32> -> tensor<2x2x1x1x32x32xf32>
  %2 = tensor.empty() : tensor<1x2x56x56x32xf32>
  %pack_1 = tensor.pack %arg2 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %2 : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%pack, %pack_0 : tensor<1x2x56x56x32xf32>, tensor<2x2x1x1x32x32xf32>) outs(%pack_1 : tensor<1x2x56x56x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %5 = arith.mulf %in, %in_2 : f32
      %6 = arith.addf %out, %5 : f32
      linalg.yield %6 : f32
  } -> tensor<1x2x56x56x32xf32>
  %unpack = tensor.unpack %3 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %arg2 : tensor<1x2x56x56x32xf32> -> tensor<1x56x56x64xf32>
    %4 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%unpack, %arg3 : tensor<1x56x56x64xf32>, tensor<64xf32>) outs(%arg2 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %5 = arith.addf %in, %in_2 : f32
      linalg.yield %5 : f32
  } -> tensor<1x56x56x64xf32>
  return %4 : tensor<1x56x56x64xf32>
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d4)>
// CHECK: func.func @conv(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]*]]: tensor<1x56x56x64xf32>,
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9]*]]: tensor<1x1x64x64xf32>,
// CHECK-SAME: %[[ARG2:[a-zA-Z0-9]*]]: tensor<1x56x56x64xf32>,
// CHECK-SAME: %[[ARG3:[a-zA-Z0-9]*]]: tensor<64xf32>) -> tensor<1x56x56x64xf32>
// CHECK: %[[BUFF0:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK0:.+]] = tensor.pack %[[ARG0]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF0]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CHECK: %[[BUFF1:.+]] = tensor.empty() : tensor<2x2x1x1x32x32xf32>
// CHECK: %[[PACK1:.+]] = tensor.pack %[[ARG1]] outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %[[BUFF1]] : tensor<1x1x64x64xf32> -> tensor<2x2x1x1x32x32xf32>
// CHECK: %[[BUFF2:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK2:.+]] = tensor.pack %[[ARG2]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF2]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CHECK: %[[VAL:.+]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%[[PACK0]], %[[PACK1]] : tensor<1x2x56x56x32xf32>, tensor<2x2x1x1x32x32xf32>) outs(%[[PACK2]] : tensor<1x2x56x56x32xf32>)
// CHECK: %[[BUFF3:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK3:.+]] = tensor.pack %[[ARG2]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF3]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CHECK: %[[EXPAND:.+]] = tensor.expand_shape %[[ARG3]] {{\[}}[0, 1]] : tensor<64xf32> into tensor<2x32xf32>
// CHECK: %[[VAL1:.+]] = linalg.generic {indexing_maps = [#[[MAP3]], #[[MAP4]], #[[MAP3]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%[[VAL]], %[[EXPAND]] : tensor<1x2x56x56x32xf32>, tensor<2x32xf32>) outs(%[[PACK3]] : tensor<1x2x56x56x32xf32>)
// CHECK: %[[UNPACK:.+]] = tensor.unpack %[[VAL1]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[ARG2]] : tensor<1x2x56x56x32xf32> -> tensor<1x56x56x64xf32>

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>

func.func @conv(%arg0: tensor<1x56x56x64xf32>, %arg1: tensor<1x1x64x64xf32>, %arg2: tensor<1x56x56x64xf32>, %arg3: tensor<56x64xf32>) -> tensor<1x56x56x64xf32> {
  %0 = tensor.empty() : tensor<1x2x56x56x32xf32>
  %pack = tensor.pack %arg0 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %0 : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
  %1 = tensor.empty() : tensor<2x2x1x1x32x32xf32>
  %pack_0 = tensor.pack %arg1 outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %1 : tensor<1x1x64x64xf32> -> tensor<2x2x1x1x32x32xf32>
  %2 = tensor.empty() : tensor<1x2x56x56x32xf32>
  %pack_1 = tensor.pack %arg2 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %2 : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%pack, %pack_0 : tensor<1x2x56x56x32xf32>, tensor<2x2x1x1x32x32xf32>) outs(%pack_1 : tensor<1x2x56x56x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %5 = arith.mulf %in, %in_2 : f32
      %6 = arith.addf %out, %5 : f32
      linalg.yield %6 : f32
  } -> tensor<1x2x56x56x32xf32>
  %unpack = tensor.unpack %3 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %arg2 : tensor<1x2x56x56x32xf32> -> tensor<1x56x56x64xf32>
  %4 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%unpack, %arg3 : tensor<1x56x56x64xf32>, tensor<56x64xf32>) outs(%arg2 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %5 = arith.addf %in, %in_2 : f32
      linalg.yield %5 : f32
  } -> tensor<1x56x56x64xf32>
  return %4 : tensor<1x56x56x64xf32>
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)> 
// CHECK: func.func @conv(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]*]]: tensor<1x56x56x64xf32>,
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9]*]]: tensor<1x1x64x64xf32>,
// CHECK-SAME: %[[ARG2:[a-zA-Z0-9]*]]: tensor<1x56x56x64xf32>,
// CHECK-SAME: %[[ARG3:[a-zA-Z0-9]*]]: tensor<56x64xf32>) -> tensor<1x56x56x64xf32>
// CHECK: %[[BUFF0:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK0:.+]] = tensor.pack %[[ARG0]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF0]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CHECK: %[[BUFF1:.+]] = tensor.empty() : tensor<2x2x1x1x32x32xf32>
// CHECK: %[[PACK1:.+]] = tensor.pack %[[ARG1]] outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %[[BUFF1]] : tensor<1x1x64x64xf32> -> tensor<2x2x1x1x32x32xf32>
// CHECK: %[[BUFF2:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK2:.+]] = tensor.pack %[[ARG2]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF2]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CHECK: %[[VAL:.+]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%[[PACK0]], %[[PACK1]] : tensor<1x2x56x56x32xf32>, tensor<2x2x1x1x32x32xf32>) outs(%[[PACK2]] : tensor<1x2x56x56x32xf32>)
// CHECK: %[[BUFF4:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK4:.+]] = tensor.pack %[[ARG2]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF4]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CHECK: %[[BUFF3:.+]] = tensor.empty() : tensor<2x56x32xf32>
// CHECK: %[[PACK3:.+]] = tensor.pack %[[ARG3]] outer_dims_perm = [1, 0] inner_dims_pos = [1] inner_tiles = [32] into %[[BUFF3]] : tensor<56x64xf32> -> tensor<2x56x32xf32>
// CHECK: %[[VAL1:.+]] = linalg.generic {indexing_maps = [#[[MAP3]], #[[MAP4]], #[[MAP3]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%[[VAL]], %[[PACK3]] : tensor<1x2x56x56x32xf32>, tensor<2x56x32xf32>) outs(%[[PACK4]] : tensor<1x2x56x56x32xf32>)
// CHECK: %[[UNPACK:.+]] = tensor.unpack %[[VAL1]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[ARG2]] : tensor<1x2x56x56x32xf32> -> tensor<1x56x56x64xf32>

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @conv(%arg0: tensor<1x56x56x64xf32>, %arg1: tensor<1x1x64x64xf32>, %arg2: tensor<1x56x56x64xf32>) -> tensor<1x58x58x64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x2x56x56x32xf32>
  %pack = tensor.pack %arg0 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %0 : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
  %1 = tensor.empty() : tensor<2x2x1x1x32x32xf32>
  %pack_0 = tensor.pack %arg1 outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %1 : tensor<1x1x64x64xf32> -> tensor<2x2x1x1x32x32xf32>
  %2 = tensor.empty() : tensor<1x2x56x56x32xf32>
  %pack_1 = tensor.pack %arg2 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %2 : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%pack, %pack_0 : tensor<1x2x56x56x32xf32>, tensor<2x2x1x1x32x32xf32>) outs(%pack_1 : tensor<1x2x56x56x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %5 = arith.mulf %in, %in_2 : f32
      %6 = arith.addf %out, %5 : f32
      linalg.yield %6 : f32
  } -> tensor<1x2x56x56x32xf32>
  %unpack = tensor.unpack %3 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %arg2 : tensor<1x2x56x56x32xf32> -> tensor<1x56x56x64xf32>
  %4 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%unpack : tensor<1x56x56x64xf32>) {
    ^bb0(%out: f32):
      %5 = arith.maxf %out, %cst : f32
      linalg.yield %5 : f32
  } -> tensor<1x56x56x64xf32>
  %padded = tensor.pad %4 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
      tensor.yield %cst : f32
  } : tensor<1x56x56x64xf32> to tensor<1x58x58x64xf32>
  return %padded : tensor<1x58x58x64xf32>
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK: func.func @conv(
// CHECK-SAME: %[[ARG0:.+]]: tensor<1x56x56x64xf32>,
// CHECK-SAME: %[[ARG1:.+]]: tensor<1x1x64x64xf32>,
// CHECK-SAME: %[[ARG2:.+]]: tensor<1x56x56x64xf32>) -> tensor<1x58x58x64xf32> {
// CHECK: %[[BUFF0:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK0:.+]] = tensor.pack %[[ARG0]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF0]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CHECK: %[[BUFF1:.+]] = tensor.empty() : tensor<2x2x1x1x32x32xf32>
// CHECK: %[[PACK1:.+]] = tensor.pack %[[ARG1]] outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %[[BUFF1]] : tensor<1x1x64x64xf32> -> tensor<2x2x1x1x32x32xf32>
// CHECK: %[[BUFF2:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK2:.+]] = tensor.pack %[[ARG2]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF2]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CHECK: %[[VAL:.+]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%[[PACK0]], %[[PACK1]] : tensor<1x2x56x56x32xf32>, tensor<2x2x1x1x32x32xf32>) outs(%[[PACK2]] : tensor<1x2x56x56x32xf32>)
// CHECK: %[[VAL1:.+]] = linalg.generic {indexing_maps = [#[[MAP3]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} outs(%[[VAL]] : tensor<1x2x56x56x32xf32>)
// CHECK: %[[PADDED:.+]] = tensor.pad %[[VAL1]] low[0, 0, 1, 1, 0] high[0, 0, 1, 1, 0] 
// CHECK: %[[OUT:.+]] = tensor.empty() : tensor<1x58x58x64xf32>
// CHECK: %[[UNPACK:.+]] = tensor.unpack %[[PADDED]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[OUT]] : tensor<1x2x58x58x32xf32> -> tensor<1x58x58x64xf32>
// CHECK: return %[[UNPACK]] : tensor<1x58x58x64xf32> 

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>

func.func @fill(%arg0: f32, %arg1: tensor<1x56x56x64xf32>, %arg2: tensor<1x1x64x64xf32>) -> tensor<1x56x56x64xf32> {
  %0 = tensor.empty() : tensor<1x56x56x64xf32>
  %1 = linalg.fill ins(%arg0 : f32) outs(%0 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %2 = tensor.empty() : tensor<1x2x56x56x32xf32>
  %pack = tensor.pack %arg1 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %2 : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
  %3 = tensor.empty() : tensor<2x2x1x1x32x32xf32>
  %pack_0 = tensor.pack %arg2 outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %3 : tensor<1x1x64x64xf32> -> tensor<2x2x1x1x32x32xf32>
  %4 = tensor.empty() : tensor<1x2x56x56x32xf32>
  %pack_1 = tensor.pack %1 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %4 : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
  %5 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%pack, %pack_0 : tensor<1x2x56x56x32xf32>, tensor<2x2x1x1x32x32xf32>) outs(%pack_1 : tensor<1x2x56x56x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %6 = arith.mulf %in, %in_2 : f32
      %7 = arith.addf %out, %6 : f32
      linalg.yield %7 : f32
  } -> tensor<1x2x56x56x32xf32>
  %unpack = tensor.unpack %5 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %1 : tensor<1x2x56x56x32xf32> -> tensor<1x56x56x64xf32>
  return %unpack : tensor<1x56x56x64xf32>
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
// CHECK: func.func @fill
// CHECK-SAME: %[[ARG0:.+]]: f32,
// CHECK-SAME: %[[ARG1:.+]]: tensor<1x56x56x64xf32>,
// CHECK-SAME: %[[ARG2:.+]]: tensor<1x1x64x64xf32>
// CHECK: %[[RES:.+]] = tensor.empty() : tensor<1x56x56x64xf32>
// CHECK: %[[EMPTY_ARG1:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK_ARG1:.+]] = tensor.pack %[[ARG1]] 
// CHECK-SAME: outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] 
// CHECK-SAME: into %[[EMPTY_ARG1]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CHECK: %[[EMPTY_ARG2:.+]] = tensor.empty() : tensor<2x2x1x1x32x32xf32>
// CHECK: %[[PACK_ARG2:.+]] = tensor.pack %[[ARG2]] 
// CHECK-SAME: outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] 
// CHECK-SAME: into %[[EMPTY_ARG2]] : tensor<1x1x64x64xf32> -> tensor<2x2x1x1x32x32xf32>
// CHECK: %[[EMPTY_FILL:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACKED_FILL:.+]] = linalg.fill ins(%[[ARG0]] : f32) 
// CHECK-SAME: outs(%[[EMPTY_FILL]] : tensor<1x2x56x56x32xf32>) -> tensor<1x2x56x56x32xf32>
// CHECK: %[[GEN:.+]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: ins(%[[PACK_ARG1]], %[[PACK_ARG2]]
// CHECK-SAME: outs(%[[PACKED_FILL]]
// CHECK: %[[UNPACK:.+]] = tensor.unpack %[[GEN]] 
// CHECK-SAME: outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] 
// CHECK-SAME: into %[[RES]] : tensor<1x2x56x56x32xf32> -> tensor<1x56x56x64xf32>
