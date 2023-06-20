// RUN: tpp-opt %s -linalg-generalize-named-ops -linalg-degeneralize-generic-ops -split-input-file | FileCheck %s

// CHECK-LABEL: degeneralize
func.func @degeneralize(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> {
  %0 = tensor.empty() : tensor<3x3xf32>
  %cst = arith.constant 0.0 : f32
  %1 = linalg.fill ins(%cst : f32) outs(%arg0 : tensor<3x3xf32>) -> tensor<3x3xf32>
  // CHECK: linalg.fill
  %2 = linalg.fill ins(%cst : f32) outs(%arg1 : tensor<3x3xf32>) -> tensor<3x3xf32>
  // CHECK-NEXT: linalg.fill
  %3 = linalg.fill ins(%cst : f32) outs(%0 : tensor<3x3xf32>) -> tensor<3x3xf32>
  // CHECK-NEXT: linalg.fill
  %4 = linalg.matmul ins(%1, %2 : tensor<3x3xf32>, tensor<3x3xf32>) 
                     outs(%3 : tensor<3x3xf32>) -> tensor<3x3xf32>
  // CHECK-NEXT: linalg.matmul
  return %4 : tensor<3x3xf32>
}

// -----

// CHECK-LABEL: degeneralize
func.func @degeneralize(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) -> memref<3x3xf32> {
  %alloc = memref.alloc() : memref<3x3xf32>
  %cst = arith.constant 0.0 : f32
  linalg.fill ins(%cst : f32) outs(%arg0 : memref<3x3xf32>)
  // CHECK: linalg.fill
  linalg.fill ins(%cst : f32) outs(%arg1 : memref<3x3xf32>)
  // CHECK-NEXT: linalg.fill
  linalg.fill ins(%cst : f32) outs(%alloc : memref<3x3xf32>)
  // CHECK-NEXT: linalg.fill
  linalg.matmul ins(%arg0, %arg1 : memref<3x3xf32>, memref<3x3xf32>) 
                outs(%alloc : memref<3x3xf32>)
  // CHECK-NEXT: linalg.matmul
  return %alloc : memref<3x3xf32>
}

// -----

func.func @degeneralize(%arg0: memref<3x3x3xf32>, %arg1: memref<3x3x3xf32>) -> memref<3x3xf32> {
  %alloc = memref.alloc() : memref<3x3xf32>
  %cst = arith.constant 0.0 : f32
  linalg.fill ins(%cst : f32) outs(%alloc : memref<3x3xf32>)
  // CHECK: linalg.fill
  linalg.batch_reduce_matmul ins(%arg0, %arg1 : memref<3x3x3xf32>, memref<3x3x3xf32>)
                             outs(%alloc : memref<3x3xf32>)
  // CHECK-NEXT: linalg.batch_reduce_matmul
  return %alloc : memref<3x3xf32>
}

// -----

func.func @degeneralize(%arg0: tensor<3x3x3xf32>, %arg1: tensor<3x3x3xf32>) -> tensor<3x3xf32> {
  %0 = tensor.empty() : tensor<3x3xf32>
  %cst = arith.constant 0.0 : f32
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<3x3xf32>) -> tensor<3x3xf32>
  // CHECK: linalg.fill
  %2 = linalg.batch_reduce_matmul ins(%arg0, %arg1 : tensor<3x3x3xf32>, tensor<3x3x3xf32>)
                                  outs(%1 : tensor<3x3xf32>) -> tensor<3x3xf32>
  // CHECK-NEXT: linalg.batch_reduce_matmul
  return %2 : tensor<3x3xf32>
}
