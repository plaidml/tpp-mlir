// RUN: tpp-opt %s -rewrite-conv-to-matmul-or-brgemm -canonicalize -split-input-file | FileCheck %s

func.func @conv2d_nhwc_hwcf_unpacked(%arg0: tensor<1x113x113x64xf32>, %arg1: tensor<3x3x64x256xf32>, %arg2: tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32> {
  %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>,
                                 strides = dense<2> : tensor<2xi64>}
    ins(%arg0, %arg1 : tensor<1x113x113x64xf32>, tensor<3x3x64x256xf32>)
    outs(%arg2: tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
  return %1 : tensor<1x56x56x256xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0 * 2 + d1)>
// CHECK: func.func @conv2d_nhwc_hwcf_unpacked
// CHECK-SAME:  %[[ARG0:.+]]: tensor<1x113x113x64xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: tensor<3x3x64x256xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32> {
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C56:.+]] = arith.constant 56 : index
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : index
// CHECK: %[[LOOP0:.+]] = scf.for %[[ARG3:.+]] = %[[C0]] to %[[C56]] step %[[C1]]
// CHECK-SAME:  iter_args(%[[ARG4:.+]] = %[[ARG2]]) -> (tensor<1x56x56x256xf32>) {
// CHECK: %[[LOOP1:.+]] = scf.for %[[ARG5:.+]] = %[[C0]] to %[[C3]] step %[[C1]]
// CHECK-SAME:  iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<1x56x56x256xf32>) {
// CHECK: %[[LOOP2:.+]] = scf.for %[[ARG7:.+]] = %[[C0]] to %[[C3]] step %[[C1]]
// CHECK-SAME:  iter_args(%[[ARG8:.+]] = %[[ARG6]]) -> (tensor<1x56x56x256xf32>) {
// CHECK: %[[APPLY:.+]] = affine.apply #[[MAP]](%[[ARG3]], %[[ARG5]])
// CHECK: %[[SLICE:.+]] = tensor.extract_slice
// CHECK-SAME:  %[[ARG0]][0, %[[APPLY]], %[[ARG7]], 0] [1, 1, 56, 64] [1, 1, 2, 1]
// CHECK-SAME:   : tensor<1x113x113x64xf32> to tensor<56x64xf32>
// CHECK: %[[SLICE0:.+]] = tensor.extract_slice
// CHECK-SAME:  %[[ARG1]][%[[ARG5]], %[[ARG7]], 0, 0] [1, 1, 64, 256] [1, 1, 1, 1]
// CHECK-SAME:  : tensor<3x3x64x256xf32> to tensor<64x256xf32>
// CHECK: %[[SLICE1:.+]] = tensor.extract_slice
// CHECK-SAME:  %[[ARG8]][0, %[[ARG3]], 0, 0] [1, 1, 56, 256] [1, 1, 1, 1]
// CHECK-SAME:  : tensor<1x56x56x256xf32> to tensor<56x256xf32>
// CHECK: %[[MUL:.+]] = linalg.matmul ins(%[[SLICE]], %[[SLICE0]] : tensor<56x64xf32>, tensor<64x256xf32>)
// CHECK-SAME:  outs(%[[SLICE1]] : tensor<56x256xf32>) -> tensor<56x256xf32>
// CHECK: %[[INSERT:.+]] = tensor.insert_slice
// CHECK-SAME:  %[[MUL]] into %[[ARG8]][0, %[[ARG3]], 0, 0] [1, 1, 56, 256] [1, 1, 1, 1]
// CHECK-SAME:  : tensor<56x256xf32> into tensor<1x56x56x256xf32>


// -----

#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>

func.func @conv_2d_blocked(%arg0: tensor<1x2x113x113x32xf32>, %arg1: tensor<8x2x3x3x32x32xf32>, %arg2: tensor<1x8x111x111x32xf32>) -> tensor<1x8x111x111x32xf32> {
  %1 = linalg.generic {
    indexing_maps = [#map3, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel",
                      "reduction", "reduction", "reduction", "reduction"]}
    ins(%arg0, %arg1: tensor<1x2x113x113x32xf32>, tensor<8x2x3x3x32x32xf32>)
    outs(%arg2: tensor<1x8x111x111x32xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %2 = arith.mulf %in, %in_1 : f32
    %3 = arith.addf %out, %2 : f32
    linalg.yield %3 : f32
  } -> tensor<1x8x111x111x32xf32>
  return %1 : tensor<1x8x111x111x32xf32>
}

// CHECK: func.func @conv_2d_blocked(
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG: %[[C111:.+]] = arith.constant 111 : index
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : index
// CHECK: %[[LOOP:.+]] = scf.for %[[ARG3:.+]] = %[[C0]] to %[[C8]] step %[[C1]]
// CHECK: %[[LOOP1:.+]] = scf.for %[[ARG5:.+]] = %[[C0]] to %[[C111]] step %[[C1]]
// CHECK: %[[LOOP2:.+]] = scf.for %[[ARG7:.+]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK: %[[LOOP3:.+]] = scf.for %[[ARG9:.+]] = %[[C0]] to %[[C3]] step %[[C1]]
// CHECK: %[[LOOP4:.+]] = scf.for %[[ARG11:.+]] = %[[C0]] to %[[C3]] step %[[C1]]
// CHECK: %[[APPLY:.+]] = affine.apply #[[MAP]](%[[ARG5]], %[[ARG9]])
// CHECK: %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]]
// CHECK-SAME:  [0, %[[ARG7]], %[[APPLY]], %[[ARG11]], 0] [1, 1, 1, 111, 32] [1, 1, 1, 1, 1]
// CHECK-SAME:  : tensor<1x2x113x113x32xf32> to tensor<111x32xf32>
// CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[ARG1]]
// CHECK-SAME:  [%[[ARG3]], %[[ARG7]], %[[ARG9]], %[[ARG11]], 0, 0] [1, 1, 1, 1, 32, 32] [1, 1, 1, 1, 1, 1]
// CHECK-SAME:  : tensor<8x2x3x3x32x32xf32> to tensor<32x32xf32>
// CHECK: %[[SLICE2:.+]] = tensor.extract_slice
// CHECK-SAME:  %{{.+}}[0, %[[ARG3]], %[[ARG5]], 0, 0] [1, 1, 1, 111, 32] [1, 1, 1, 1, 1]
// CHECK-SAME:  : tensor<1x8x111x111x32xf32> to tensor<111x32xf32>
// CHECK: %[[MUL:.+]] = linalg.matmul
// CHECK-SAME:  ins(%[[SLICE]], %[[SLICE1]] : tensor<111x32xf32>, tensor<32x32xf32>)
// CHECK-SAME:  outs(%[[SLICE2]] : tensor<111x32xf32>) -> tensor<111x32xf32>
// CHECK: {{.+}} = tensor.insert_slice %[[MUL]]
// CHECK-SAME:  into %{{.+}}[0, %[[ARG3]], %[[ARG5]], 0, 0] [1, 1, 1, 111, 32] [1, 1, 1, 1, 1]
// CHECK-SAME:  : tensor<111x32xf32> into tensor<1x8x111x111x32xf32>
