// RUN: tpp-opt %s -linalg-generalize-named-ops -linalg-degeneralize-generic-ops -split-input-file | FileCheck %s

func.func @degeneralize(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> {
  %0 = tensor.empty() : tensor<3x3xf32>
  %cst = arith.constant 0.0 : f32
  %1 = linalg.fill ins(%cst : f32) outs(%arg0 : tensor<3x3xf32>) -> tensor<3x3xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%arg1 : tensor<3x3xf32>) -> tensor<3x3xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%0 : tensor<3x3xf32>) -> tensor<3x3xf32>
  %4 = linalg.matmul ins(%1, %2 : tensor<3x3xf32>, tensor<3x3xf32>)
                     outs(%3 : tensor<3x3xf32>) -> tensor<3x3xf32>
  return %4 : tensor<3x3xf32>
}

// CHECK-LABEL: degeneralize
// CHECK-SAME: %[[ARG0:.+]]: tensor<3x3xf32>, %[[ARG1:.+]]: tensor<3x3xf32>
// CHECK: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<3x3xf32>
// CHECK: %[[FILL_ARG0:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[ARG0]] : tensor<3x3xf32>) -> tensor<3x3xf32>
// CHECK: %[[FILL_ARG1:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[ARG1]] : tensor<3x3xf32>) -> tensor<3x3xf32>
// CHECK: %[[OUT:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EMPTY]] : tensor<3x3xf32>) -> tensor<3x3xf32>
// CHECK: %{{.+}} = linalg.matmul ins(%[[FILL_ARG0]], %[[FILL_ARG1]] : tensor<3x3xf32>, tensor<3x3xf32>)
// CHECK-SAME:  outs(%[[OUT]] : tensor<3x3xf32>) -> tensor<3x3xf32>

// -----

func.func @degeneralize(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) -> memref<3x3xf32> {
  %alloc = memref.alloc() : memref<3x3xf32>
  %cst = arith.constant 0.0 : f32
  linalg.fill ins(%cst : f32) outs(%arg0 : memref<3x3xf32>)
  linalg.fill ins(%cst : f32) outs(%arg1 : memref<3x3xf32>)
  linalg.fill ins(%cst : f32) outs(%alloc : memref<3x3xf32>)
  linalg.matmul ins(%arg0, %arg1 : memref<3x3xf32>, memref<3x3xf32>)
                outs(%alloc : memref<3x3xf32>)
  return %alloc : memref<3x3xf32>
}

// CHECK-LABEL: degeneralize
// CHECK-SAME:  %[[ARG0:.+]]: memref<3x3xf32>, %[[ARG1:.+]]: memref<3x3xf32>
// CHECK: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[ALLOC:.+]] = memref.alloc() : memref<3x3xf32>
// CHECK: linalg.fill ins(%[[CST]] : f32) outs(%[[ARG0]] : memref<3x3xf32>)
// CHECK: linalg.fill ins(%[[CST]] : f32) outs(%[[ARG1]] : memref<3x3xf32>)
// CHECK: linalg.fill ins(%[[CST]] : f32) outs(%[[ALLOC]] : memref<3x3xf32>)
// CHECK: linalg.matmul ins(%[[ARG0]], %[[ARG1]] : memref<3x3xf32>, memref<3x3xf32>)
// CHECK-SAME:  outs(%[[ALLOC]] : memref<3x3xf32>)

// -----

func.func @degeneralize(%arg0: memref<3x3x3xf32>, %arg1: memref<3x3x3xf32>) -> memref<3x3xf32> {
  %alloc = memref.alloc() : memref<3x3xf32>
  %cst = arith.constant 0.0 : f32
  linalg.fill ins(%cst : f32) outs(%alloc : memref<3x3xf32>)
  linalg.batch_reduce_matmul ins(%arg0, %arg1 : memref<3x3x3xf32>, memref<3x3x3xf32>)
                             outs(%alloc : memref<3x3xf32>)
  return %alloc : memref<3x3xf32>
}

// CHECK-LABEL: degeneralize
// CHECK-SAME: %[[ARG0:.+]]: memref<3x3x3xf32>, %[[ARG1:.+]]: memref<3x3x3xf32>
// CHECK: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[ALLOC:.+]] = memref.alloc() : memref<3x3xf32>
// CHECK: linalg.fill ins(%[[CST]] : f32) outs(%[[ALLOC]] : memref<3x3xf32>)
// CHECK: linalg.batch_reduce_matmul ins(%[[ARG0]], %[[ARG1]] : memref<3x3x3xf32>, memref<3x3x3xf32>)
// CHECK-SAME:  outs(%[[ALLOC]] : memref<3x3xf32>)

// -----

func.func @degeneralize(%arg0: tensor<3x3x3xf32>, %arg1: tensor<3x3x3xf32>) -> tensor<3x3xf32> {
  %0 = tensor.empty() : tensor<3x3xf32>
  %cst = arith.constant 0.0 : f32
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<3x3xf32>) -> tensor<3x3xf32>
  %2 = linalg.batch_reduce_matmul ins(%arg0, %arg1 : tensor<3x3x3xf32>, tensor<3x3x3xf32>)
                                  outs(%1 : tensor<3x3xf32>) -> tensor<3x3xf32>
  return %2 : tensor<3x3xf32>
}

// CHECK: degeneralize
// CHECK-SAME:  %[[ARG0:.+]]: tensor<3x3x3xf32>, %[[ARG1:.+]]: tensor<3x3x3xf32>
// CHECK: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<3x3xf32>
// CHECK: %[[FILL:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EMPTY]] : tensor<3x3xf32>) -> tensor<3x3xf32>
// CHECK: %{{.+}} = linalg.batch_reduce_matmul ins(%[[ARG0]], %[[ARG1]] : tensor<3x3x3xf32>, tensor<3x3x3xf32>)
// CHECK-SAME:  outs(%[[FILL]] : tensor<3x3xf32>) -> tensor<3x3xf32>
