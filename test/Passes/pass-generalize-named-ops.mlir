// RUN: tpp-opt %s -generalize-named-ops -split-input-file | FileCheck %s

func.func @generalize_matmul(%A : memref<16x8xf32>, %B: memref<8x32xf32>, %C: memref<16x32xf32>) {
  linalg.matmul ins(%A, %B: memref<16x8xf32>, memref<8x32xf32>)
                outs(%C: memref<16x32xf32>)
  return
}

// CHECK-LABEL @generalize_matmul(
// CHECK-NOT: linalg.matmul
// CHECK: linalg.generic

// -----

func.func @generalize_add(%A : memref<32x32xf32>, %B: memref<32x32xf32>, %C: memref<32x32xf32>) {
  linalg.add ins(%A, %B: memref<32x32xf32>, memref<32x32xf32>)
             outs(%C: memref<32x32xf32>)
  return
}

// CHECK-LABEL @generalize_add(
// CHECK-NOT: linalg.add
// CHECK: linalg.generic

// -----

func.func @dont_generalize_fill(%arg0 : memref<32x32xf32>, %val: f32) {
  linalg.fill ins(%val : f32) outs(%arg0 : memref<32x32xf32>)
  return
}

// CHECK-LABEL @generalize_add(
// CHECK: linalg.fill
// CHECK-NOT: linalg.generic
