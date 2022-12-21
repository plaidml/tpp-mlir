// RUN: tpp-opt %s -generalize-tensor-pack-unpack -split-input-file | FileCheck %s
// XFAIL: *

func.func @simple_KCRS_to_KCRSsr(%arg0: tensor<1x1x32x8xf32>, %arg1: tensor<1x1x1x1x8x32xf32>) -> tensor<1x1x1x1x8x32xf32> {
  %0 = tensor.pack %arg0 inner_dims_pos = [3, 2] inner_tiles = [8, 32] into %arg1 : tensor<1x1x32x8xf32> -> tensor<1x1x1x1x8x32xf32>
  return %0 : tensor<1x1x1x1x8x32xf32>
}
// CHECK-LABEL: func.func @simple_KCRS_to_KCRSsr
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK:         %[[TILE:.+]] = tensor.extract_slice %[[SRC]][0, 0, 0, 0] [1, 1, 32, 8] [1, 1, 1, 1]
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<8x32xf32>
// CHECK:         %[[TRANSP:.+]] =  linalg.transpose
// CHECK-SAME:      ins(%[[TILE]] : tensor<32x8xf32>)
// CHECK-SAME:      outs(%[[EMPTY]] : tensor<8x32xf32>)
// CHECK-SAME:      permutation = [1, 0]
// CHECK:         %[[INSERT:.+]] = tensor.insert_slice %[[TRANSP]] into %[[DEST]]
// CHECK-SAME:      [0, 0, 0, 0, 0, 0] [1, 1, 1, 1, 8, 32] [1, 1, 1, 1, 1, 1]
// CHECK:         return %[[INSERT]]

// -----

func.func @KC_to_CKkc(%arg0: tensor<128x256xf32>, %arg1: tensor<32x4x32x8xf32>) -> tensor<32x4x32x8xf32> {
  %0 = tensor.pack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 8] into %arg1 : tensor<128x256xf32> -> tensor<32x4x32x8xf32>
  return %0 : tensor<32x4x32x8xf32>
}
// CHECK-DAG:   #[[MAP0:.+]] = affine_map<(d0) -> (d0 * 32)>
// CHECK-DAG:   #[[MAP1:.+]] = affine_map<(d0) -> (32, d0 * -32 + 128)>
// CHECK-DAG:   #[[MAP2:.+]] = affine_map<(d0) -> (d0 * 8)>
// CHECK-DAG:   #[[MAP3:.+]] = affine_map<(d0) -> (8, d0 * -8 + 256)>
// CHECK:       func.func @KC_to_CKkc
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK:         %{{.+}} = scf.for %[[C:[a-zA-Z0-9]+]] =
// CHECK:           %{{.+}} = scf.for %[[K:[a-zA-Z0-9]+]] =
// CHECK-DAG:         %[[IN_K:.+]] = affine.apply #[[MAP0]](%[[K]])
// CHECK-DAG:         %[[IN_K_SZ:.+]] = affine.min #[[MAP1]](%[[K]])
// CHECK-DAG:         %[[IN_C:.+]] = affine.apply #[[MAP2]](%[[C]])
// CHECK-DAG:         %[[IN_C_SZ:.+]] = affine.min #[[MAP3]](%[[C]])
// CHECK:             %[[SRC_SLICE:.+]] = tensor.extract_slice %[[SRC]]
// CHECK-SAME:          [%[[IN_K]], %[[IN_C]]] [%[[IN_K_SZ]], %[[IN_C_SZ]]] [1, 1]
// CHECK:             %[[TILE:.+]] = tensor.extract_slice %[[SRC_SLICE]]
// CHECK-SAME:          [0, 0] [32, 8] [1, 1] : tensor<?x?xf32> to tensor<32x8xf32>
// CHECK:             %[[EMPTY:.+]] = tensor.empty() : tensor<32x8xf32>
// CHECK:             %[[TRANSP:.+]] =  linalg.transpose
// CHECK-SAME:          ins(%[[TILE]]
// CHECK-SAME:          outs(%[[EMPTY]]
// CHECK-SAME:          permutation = [0, 1]
// CHECK:             %[[SUB_ITER:.+]] = tensor.insert_slice %[[TRANSP]] into %{{[a-zA-Z0-9]+}}
// CHECK-SAME:          [0, 0, 0, 0] [1, 1, 32, 8] [1, 1, 1, 1] : tensor<32x8xf32> into tensor<1x1x32x8xf32>
// CHECK:             %{{.+}} = tensor.insert_slice %[[SUB_ITER]] into %{{[a-zA-Z0-9]+}}
// CHECK-SAME:          [%[[C]], %[[K]], 0, 0] [1, 1, 32, 8] [1, 1, 1, 1] : tensor<1x1x32x8xf32> into tensor<32x4x32x8xf32>

// -----

func.func @simple_KCRSsr_to_KCRS(%arg0: tensor<1x1x1x1x8x32xf32>, %arg1: tensor<1x1x32x8xf32>) -> tensor<1x1x32x8xf32> {
  %0 = tensor.unpack %arg0 inner_dims_pos = [3, 2] inner_tiles = [8, 32] into %arg1 : tensor<1x1x1x1x8x32xf32> -> tensor<1x1x32x8xf32>
  return %0 : tensor<1x1x32x8xf32>
}
// CHECK-LABEL: func.func @simple_KCRSsr_to_KCRS
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK:         %[[TILE:.+]] = tensor.extract_slice %[[SRC]][0, 0, 0, 0, 0, 0] [1, 1, 1, 1, 8, 32] [1, 1, 1, 1, 1, 1]
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<32x8xf32>
// CHECK:         %[[TRANSP:.+]] =  linalg.transpose
// CHECK-SAME:      ins(%[[TILE]] : tensor<8x32xf32>)
// CHECK-SAME:      outs(%[[EMPTY]] : tensor<32x8xf32>)
// CHECK-SAME:      permutation = [1, 0]
// CHECK:         %[[INSERT:.+]] = tensor.insert_slice %[[TRANSP]] into %[[DEST]]
// CHECK-SAME:      [0, 0, 0, 0] [1, 1, 32, 8] [1, 1, 1, 1]
// CHECK:         return %[[INSERT]]

// -----

func.func @unpack_and_extract_slice(%arg0: tensor<2x8x8x2xf32>, %arg1: tensor<13x15xf32>) -> tensor<13x15xf32> {
  %0 = tensor.unpack %arg0 inner_dims_pos = [0, 1] inner_tiles = [8, 2] into %arg1 : tensor<2x8x8x2xf32> -> tensor<13x15xf32>
  return %0 : tensor<13x15xf32>
}
// CHECK-DAG:   #[[MAP0:.+]] = affine_map<(d0) -> (8, -d0 + 13)>
// CHECK-DAG:   #[[MAP1:.+]] = affine_map<(d0) -> (2, -d0 + 15)>
// CHECK-DAG:   #[[MAP2:.+]] = affine_map<(d0) -> (d0 floordiv 8)>
// CHECK-DAG:   #[[MAP3:.+]] = affine_map<(d0) -> (d0 floordiv 2)>
// CHECK:       func.func @unpack_and_extract_slice
// CHECK-SAME:    %[[SRC:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[DEST:[a-zA-Z0-9]+]]
// CHECK:         %{{.+}} = scf.for %[[I:[a-zA-Z0-9]+]] =
// CHECK:           %[[OUT_I_SZ:.+]] = affine.min #[[MAP0]](%[[I]])
// CHECK:           %{{.+}} = scf.for %[[J:[a-zA-Z0-9]+]] =
// CHECK:             %[[OUT_J_SZ:.+]] = affine.min #[[MAP1]](%[[J]])
// CHECK:             %[[IN_I:.+]] = affine.apply #[[MAP2]](%[[I]])
// CHECK:             %[[IN_J:.+]] = affine.apply #[[MAP3]](%[[J]])
// CHECK:             %[[SRC_SLICE:.+]] = tensor.extract_slice %[[SRC]]
// CHECK-SAME:          [%[[IN_I]], %[[IN_J]], 0, 0] [1, 1, 8, 2] [1, 1, 1, 1]
// CHECK:             %[[ITER_SLICE:.+]] = tensor.extract_slice %{{[a-zA-Z0-9]+}}
// CHECK-SAME:          [%[[I]], %[[J]]] [%[[OUT_I_SZ]], %[[OUT_J_SZ]]]
// CHECK:             %[[TILE:.+]] = tensor.extract_slice %[[SRC_SLICE]]
// CHECK-SAME:          [0, 0, 0, 0] [1, 1, 8, 2] [1, 1, 1, 1] : tensor<1x1x8x2xf32> to tensor<8x2xf32>
// CHECK:             %[[EMPTY:.+]] = tensor.empty() : tensor<8x2xf32>
// CHECK:             %[[TRANSP:.+]] =  linalg.transpose
// CHECK-SAME:          ins(%[[TILE]] : tensor<8x2xf32>)
// CHECK-SAME:          outs(%[[EMPTY]] : tensor<8x2xf32>)
// CHECK-SAME:          permutation = [0, 1]
// CHECK:             %[[UNPACK_TILE:.+]] = tensor.extract_slice %[[TRANSP]]
// CHECK-SAME:          [0, 0] [%[[OUT_I_SZ]], %[[OUT_J_SZ]]] [1, 1]
// CHECK:             %[[INSERT1:.+]] = tensor.insert_slice %[[UNPACK_TILE]] into %[[ITER_SLICE]]
// CHECK-SAME:          [0, 0] [%[[OUT_I_SZ]], %[[OUT_J_SZ]]] [1, 1]
// CHECK:             %[[INSERT2:.+]] = tensor.insert_slice %[[INSERT1]] into %{{[a-zA-Z0-9]+}}
// CHECK-SAME:          [%[[I]], %[[J]]] [%[[OUT_I_SZ]], %[[OUT_J_SZ]]] [1, 1]

// -----

//func.func @pack_conv(%arg0: tensor<1x6x6x8xf32>, %arg1: tensor<1x4x6x6x2xf32>) -> tensor<1x4x6x6x2xf32> {
//  %0 = tensor.pack %arg0 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [2] into %arg1 : tensor<1x6x6x8xf32> -> tensor<1x4x6x6x2xf32>
//  return %0 : tensor<1x4x6x6x2xf32>
//}

// -----

//func.func @pack_conv(%arg0: tensor<1x1x8x8xf32>, %arg1: tensor<4x4x1x1x2x2xf32>) -> tensor<4x4x1x1x2x2xf32> {
//  %0 = tensor.pack %arg0 outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [2, 2] into %arg1 : tensor<1x1x8x8xf32> -> tensor<4x4x1x1x2x2xf32>
//  return %0 : tensor<4x4x1x1x2x2xf32>
//}

// -----

//func.func @unpack_conv(%arg0: tensor<1x4x6x6x2xf32>, %arg1: tensor<1x6x6x8xf32>) -> tensor<1x6x6x8xf32> {
//  %0 = tensor.unpack %arg0 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [2] into %arg1 : tensor<1x4x6x6x2xf32> -> tensor<1x6x6x8xf32>
//  return %0 : tensor<1x6x6x8xf32>
//}

// -----

//func.func @KCRSsr_to_KCRS(%arg0: tensor<1x1x4x8x8x32xf32>, %arg1: tensor<1x1x128x64xf32>) -> tensor<1x1x128x64xf32> {
//  %0 = tensor.unpack %arg0 inner_dims_pos = [3, 2] inner_tiles = [8, 32] into %arg1 : tensor<1x1x4x8x8x32xf32> -> tensor<1x1x128x64xf32>
//  return %0 : tensor<1x1x128x64xf32>
//}
