// RUN: tpp-opt %s -split-reduction-dim=tile=8 -split-input-file | FileCheck %s

func.func @tile_matmul(%A: memref<32x64xf32>, %B: memref<64x16xf32>, %C: memref<32x16xf32>) {
  linalg.matmul ins(%A, %B: memref<32x64xf32>, memref<64x16xf32>) outs(%C: memref<32x16xf32>)
  return
}

// CHECK-LABEL: @tile_matmul(
// CHECK-SAME:  %[[A:[0-9a-z]+]]: memref<32x64xf32>
// CHECK-SAME:  %[[B:[0-9a-z]+]]: memref<64x16xf32>
// CHECK-SAME:  %[[C:[0-9a-z]+]]: memref<32x16xf32>
// CHECK-DAG: %[[K_TILE:.+]] = arith.constant 8 : index
// CHECK: scf.for %[[IV:.+]] ={{.*}}step %[[K_TILE]]
// CHECK:   %[[SUBVIEW_A:.+]] = memref.subview %[[A]][0, %[[IV]]] [32, 8] [1, 1]
// CHECK:   %[[SUBVIEW_B:.+]] = memref.subview %[[B]][%[[IV]], 0] [8, 16] [1, 1]
// CHECK:   linalg.matmul ins(%[[SUBVIEW_A]], %[[SUBVIEW_B]]
// CHECK-SAME: outs(%[[C]]

// -----

func.func @no_tile_tensor(%A: tensor<32x64xf32>, %B: tensor<64x16xf32>, %C: tensor<32x16xf32>)
    -> tensor<32x16xf32> {
  %0 = linalg.matmul ins(%A, %B: tensor<32x64xf32>, tensor<64x16xf32>)
    outs(%C: tensor<32x16xf32>) -> tensor<32x16xf32>
  return %0 : tensor<32x16xf32>
}

// CHECK-LABEL: @no_tile_tensor
// CHECK-NOT: scf.for
// CHECK:   linalg.matmul

// -----

func.func @no_tile_reduction_not_divisible(%A: memref<32x60xf32>, %B: memref<60x16xf32>, %C: memref<32x16xf32>) {
  linalg.matmul ins(%A, %B: memref<32x60xf32>, memref<60x16xf32>) outs(%C: memref<32x16xf32>)
  return
}

// CHECK-LABEL: @no_tile_reduction_not_divisible
// CHECK-NOT: scf.for
// CHECK:   linalg.matmul

// -----

func.func @no_tile_dynamic(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) {
  linalg.matmul ins(%A, %B: memref<?x?xf32>, memref<?x?xf32>) outs(%C: memref<?x?xf32>)
  return
}

// CHECK-LABEL: @no_tile_dynamic
// CHECK-NOT: scf.for
// CHECK:   linalg.matmul
