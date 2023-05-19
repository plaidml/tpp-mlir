//RUN: tpp-opt %s -tpp-combine -split-input-file | FileCheck %s

func.func @fused_brgemm_test0(%arg0: tensor<4x32x32xf32>, %arg1: tensor<4x32x32xf32>, %arg2: tensor<32x32xf32>,
                        %arg3: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = tpp.brgemm (%arg0: tensor<4x32x32xf32>, %arg1: tensor<4x32x32xf32>, %arg2: tensor<32x32xf32>) -> tensor<32x32xf32>
  %1 = tpp.add (%0: tensor<32x32xf32>, %arg3: tensor<32x32xf32>) -> tensor<32x32xf32>
  %2 = tpp.relu (%1: tensor<32x32xf32>) -> tensor<32x32xf32>
  return %2 : tensor<32x32xf32>
}

// CHECK-LABEL: fused_brgemm_test0
// CHECK-SAME: %[[ARG0:.+]]: tensor<4x32x32xf32>, %[[ARG1:.+]]: tensor<4x32x32xf32>, 
// CHECK-SAME: %[[ARG2:.+]]: tensor<32x32xf32>, %[[ARG3:.+]]: tensor<32x32xf32>
// CHECK: {{.+}} = tpp.fused_brgemm [unary = relu, binary = add]
// CHECK-SAME: (%[[ARG0]] : tensor<4x32x32xf32>, %[[ARG1]] : tensor<4x32x32xf32>, %[[ARG2]] : tensor<32x32xf32>, %[[ARG3]] : tensor<32x32xf32>) -> (tensor<32x32xf32>)

// -----

func.func @fused_brgemm_test1(%arg0: tensor<4x32x32xf32>, %arg1: tensor<4x32x32xf32>, %arg2: tensor<32x32xf32>,
                        %arg3: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = tpp.brgemm (%arg0: tensor<4x32x32xf32>, %arg1: tensor<4x32x32xf32>, %arg2: tensor<32x32xf32>) -> tensor<32x32xf32>
  %1 = tpp.add (%arg3: tensor<32x32xf32>, %0: tensor<32x32xf32>) -> tensor<32x32xf32>
  %2 = tpp.relu (%1: tensor<32x32xf32>) -> tensor<32x32xf32>
  return %2 : tensor<32x32xf32>
}

// CHECK-LABEL: fused_brgemm_test1
// CHECK-SAME: %[[ARG0:.+]]: tensor<4x32x32xf32>, %[[ARG1:.+]]: tensor<4x32x32xf32>, 
// CHECK-SAME: %[[ARG2:.+]]: tensor<32x32xf32>, %[[ARG3:.+]]: tensor<32x32xf32>
// CHECK: {{.+}} = tpp.fused_brgemm [unary = relu, binary = add]
// CHECK-SAME: (%[[ARG0]] : tensor<4x32x32xf32>, %[[ARG1]] : tensor<4x32x32xf32>, %[[ARG2]] : tensor<32x32xf32>, %[[ARG3]] : tensor<32x32xf32>) -> (tensor<32x32xf32>)
