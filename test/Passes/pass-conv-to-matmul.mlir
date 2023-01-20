// RUN: tpp-opt %s -rewrite-conv-to-matmul-or-brgemm -canonicalize -split-input-file | FileCheck %s

func.func @conv2d_stride(%arg0: tensor<1x113x113x64xf32>, %arg1: tensor<3x3x64x256xf32>, %arg2: tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32> {
  %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>,
                                 strides = dense<2> : tensor<2xi64>}
    ins(%arg0, %arg1 : tensor<1x113x113x64xf32>, tensor<3x3x64x256xf32>)
    outs(%arg2: tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32>
  return %1 : tensor<1x56x56x256xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0 * 2 + d1)>
// CHECK: func.func @conv2d_stride
// CHECK-SAME:  %[[ARG0:.+]]: tensor<1x113x113x64xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: tensor<3x3x64x256xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: tensor<1x56x56x256xf32>) -> tensor<1x56x56x256xf32> {
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C56:.+]] = arith.constant 56 : index
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : index
// CHECK: %[[LOOP0:.+]] = scf.for %[[ARG3:.+]] = %[[C0]] to %[[C56]] step %[[C1]] iter_args(%[[ARG4:.+]] = %[[ARG2]]) -> (tensor<1x56x56x256xf32>) {
// CHECK: %[[LOOP1:.+]] = scf.for %[[ARG5:.+]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<1x56x56x256xf32>) {
// CHECK: %[[LOOP2:.+]] = scf.for %[[ARG7:.+]] = %[[C0]] to %[[C3]] step %[[C1]] iter_args(%[[ARG8:.+]] = %[[ARG6]]) -> (tensor<1x56x56x256xf32>) {
// CHECK: %[[APPLY:.+]] = affine.apply #[[MAP]](%[[ARG3]], %[[ARG5]])
// CHECK: %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]][0, %[[APPLY]], %[[ARG7]], 0] [1, 1, 56, 64] [1, 1, 2, 1] : tensor<1x113x113x64xf32> to tensor<56x64xf32>
// CHECK: %[[SLICE0:.+]] = tensor.extract_slice %[[ARG1]][%[[ARG5]], %[[ARG7]], 0, 0] [1, 1, 64, 256] [1, 1, 1, 1] : tensor<3x3x64x256xf32> to tensor<64x256xf32>
// CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[ARG8]][0, %[[ARG3]], 0, 0] [1, 1, 56, 256] [1, 1, 1, 1] : tensor<1x56x56x256xf32> to tensor<56x256xf32>
// CHECK: %[[MUL:.+]] = linalg.matmul ins(%[[SLICE]], %[[SLICE0]] : tensor<56x64xf32>, tensor<64x256xf32>) outs(%[[SLICE1]] : tensor<56x256xf32>) -> tensor<56x256xf32>
// CHECK: %[[INSERT:.+]] = tensor.insert_slice %[[MUL]] into %[[ARG8]][0, %[[ARG3]], 0, 0] [1, 1, 56, 256] [1, 1, 1, 1] : tensor<56x256xf32> into tensor<1x56x56x256xf32>
