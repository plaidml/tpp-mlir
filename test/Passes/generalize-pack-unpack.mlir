// RUN: tpp-opt %s -generalize-tensor-pack-unpack="convert-to-linalg=true" -split-input-file | FileCheck %s

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

func.func @unpack_conv(%arg0: tensor<1x4x6x6x2xf32>, %arg1: tensor<1x6x6x8xf32>) -> tensor<1x6x6x8xf32> {
  %0 = tensor.unpack %arg0 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [2] into %arg1 : tensor<1x4x6x6x2xf32> -> tensor<1x6x6x8xf32>
  return %0 : tensor<1x6x6x8xf32>
}

// CHECK: func.func @unpack_conv(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<1x4x6x6x2xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: tensor<1x6x6x8xf32>)
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<1x6x6x4x2xf32>
// CHECK: %[[TRANSPOSE:.+]] = linalg.transpose ins(%[[ARG0]] : tensor<1x4x6x6x2xf32>) outs(%[[EMPTY]] : tensor<1x6x6x4x2xf32>) permutation = [0, 2, 3, 1, 4]
// CHECK: %[[COLLAPSED:.+]] = tensor.collapse_shape %[[TRANSPOSE]] {{\[}}[0], [1], [2], [3, 4]] : tensor<1x6x6x4x2xf32> into tensor<1x6x6x8xf32>
// CHECK: %[[COPY:.+]] = linalg.copy ins(%[[COLLAPSED]] : tensor<1x6x6x8xf32>) outs(%[[ARG1]] : tensor<1x6x6x8xf32>) -> tensor<1x6x6x8xf32>

// -----

func.func @KCRSsr_to_KCRS(%arg0: tensor<1x1x4x8x8x32xf32>, %arg1: tensor<1x1x128x64xf32>) -> tensor<1x1x128x64xf32> {
  %0 = tensor.unpack %arg0 inner_dims_pos = [3, 2] inner_tiles = [8, 32] into %arg1 : tensor<1x1x4x8x8x32xf32> -> tensor<1x1x128x64xf32>
  return %0 : tensor<1x1x128x64xf32>
}

// CHECK: func.func @KCRSsr_to_KCRS(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<1x1x4x8x8x32xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: tensor<1x1x128x64xf32>)
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<1x1x4x32x8x8xf32>
// CHECK: %[[TRANSPOSE:.+]] = linalg.transpose ins(%[[ARG0]] : tensor<1x1x4x8x8x32xf32>) outs(%[[EMPTY]] : tensor<1x1x4x32x8x8xf32>) permutation = [0, 1, 2, 5, 3, 4]
// CHECK: %[[COLLAPSED:.+]] = tensor.collapse_shape %[[TRANSPOSE]] {{\[}}[0], [1], [2, 3], [4, 5]] : tensor<1x1x4x32x8x8xf32> into tensor<1x1x128x64xf32>
// CHECK: %[[COPY:.+]] = linalg.copy ins(%[[COLLAPSED]] : tensor<1x1x128x64xf32>) outs(%[[ARG1]] : tensor<1x1x128x64xf32>) -> tensor<1x1x128x64xf32>
