// RUN: tpp-opt %s -fuse-pack-consumer-producer -canonicalize | FileCheck %s

func.func @pack_fusion(%arg0: tensor<1024x512xbf16>, %arg1: tensor<16x32x16x32x2xbf16>) -> tensor<16x32x16x32x2xbf16> {
  %1 = tensor.empty() : tensor<16x32x32x32xbf16>
  %pack_0 = tensor.pack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %1 
    : tensor<1024x512xbf16> -> tensor<16x32x32x32xbf16>
  %pack_1 = tensor.pack %pack_0 inner_dims_pos = [2] inner_tiles = [2] into %arg1 
    : tensor<16x32x32x32xbf16> -> tensor<16x32x16x32x2xbf16>
  return %pack_1 : tensor<16x32x16x32x2xbf16>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0) -> (d0 * 32)>

// CHECK-LABEL: pack_fusion
// CHECK-SAME: %[[ARG0:.+]]: tensor<1024x512xbf16>, %[[ARG1:.+]]: tensor<16x32x16x32x2xbf16>
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<16x32x32x32xbf16>
// CHECK: %{{.+}} = scf.forall (%[[ARG2:.+]], %[[ARG3:.+]]) in (16, 32) shared_outs(%[[ARG4:.+]] = %[[ARG1]])
// CHECK: %[[AFFINE_APPLY:.+]] = affine.apply #[[MAP]](%[[ARG3]])
// CHECK: %[[AFFINE_APPLY_1:.+]] = affine.apply #[[MAP]](%[[ARG2]])
// CHECK: %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[AFFINE_APPLY]], %[[AFFINE_APPLY_1]]] [32, 32] [1, 1] 
// CHECK-SAME:  : tensor<1024x512xbf16> to tensor<32x32xbf16>
// CHECK: %[[SLICE_1:.+]] = tensor.extract_slice %[[EMPTY]][%[[ARG2]], %[[ARG3]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] 
// CHECK-SAME:  : tensor<16x32x32x32xbf16> to tensor<1x1x32x32xbf16>
// CHECK: %[[PACK:.+]] = tensor.pack %[[SLICE]] outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] 
// CHECK-SAME:  into %[[SLICE_1]] : tensor<32x32xbf16> -> tensor<1x1x32x32xbf16>
// CHECK: %[[SLICE_2:.+]] = tensor.extract_slice %[[ARG4]][%[[ARG2]], %[[ARG3]], 0, 0, 0] [1, 1, 16, 32, 2] [1, 1, 1, 1, 1] 
// CHECK-SAME:  : tensor<16x32x16x32x2xbf16> to tensor<1x1x16x32x2xbf16>
// CHECK: %[[PACK_1:.+]] = tensor.pack %[[PACK]] inner_dims_pos = [2] inner_tiles = [2] 
// CHECK-SAME:  into %[[SLICE_2]] : tensor<1x1x32x32xbf16> -> tensor<1x1x16x32x2xbf16>
